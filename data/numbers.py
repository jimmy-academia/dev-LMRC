import random
import os
import time
from datetime import datetime

from utils import dumpj, create_llm_client, user_struct

# --- Configuration ---
NUM_ITEMS_TO_GENERATE = 200  # Total number of items
OUTPUT_FILENAME = f"3c_product_data.json"
LLM_REQUEST_DELAY_SECONDS = 2 # Delay between LLM calls to avoid rate limits

# --- Product Specification Templates with Tiers ---

spec_templates = {
    "Laptop": {
        "Budget": {
            "price": (400, 700), "ram_gb": [8], "storage_gb": [256, 512],
            "cpu": ["Intel Core i3", "AMD Ryzen 3"], "gpu": ["Integrated Intel UHD/Iris Xe", "Integrated AMD Radeon"],
            "screen_inches": (14.0, 15.6), "resolution": ["1920x1080"], "refresh_hz": [60],
            "battery_hours": (4, 7), "weight_kg": (1.4, 2.0)
        },
        "Mid-Range": {
            "price": (700, 1200), "ram_gb": [16], "storage_gb": [512, 1000],
            "cpu": ["Intel Core i5", "AMD Ryzen 5"], "gpu": ["Integrated Intel Iris Xe", "NVIDIA MX550", "Basic NVIDIA RTX 3050/4050"],
            "screen_inches": (14.0, 16.0), "resolution": ["1920x1080", "1920x1200", "2560x1440"], "refresh_hz": [60, 90, 120],
            "battery_hours": (6, 10), "weight_kg": (1.2, 1.8)
        },
        "Premium/Ultrabook": {
            "price": (1100, 2500), "ram_gb": [16, 32], "storage_gb": [512, 1000, 2000],
            "cpu": ["Intel Core i7/i9", "AMD Ryzen 7/9"], "gpu": ["Integrated Intel Iris Xe", "Integrated AMD Radeon Graphics"],
            "screen_inches": (13.0, 15.0), "resolution": ["1920x1200", "2560x1600", "2880x1800", "3840x2160"], "refresh_hz": [60, 90, 120],
            "battery_hours": (8, 15), "weight_kg": (0.9, 1.4)
        },
        "Gaming": {
            "price": (1000, 3500), "ram_gb": [16, 32, 64], "storage_gb": [1000, 2000, 4000],
            "cpu": ["Intel Core i7/i9 HX", "AMD Ryzen 7/9 HX"], "gpu": ["NVIDIA RTX 4060", "NVIDIA RTX 4070", "NVIDIA RTX 4080/4090", "AMD RX 7700S/7800M"],
            "screen_inches": (15.6, 17.3), "resolution": ["1920x1080", "2560x1440"], "refresh_hz": [144, 165, 240, 300],
            "battery_hours": (3, 6), "weight_kg": (1.8, 3.0) # Lower battery life during gaming
        }
    },
    "Smartphone": {
        "Budget": {
            "price": (150, 400), "ram_gb": [4, 6], "storage_gb": [64, 128],
            "screen_inches": (6.2, 6.6), "resolution": ["HD+ (e.g. 1600x720)", "FHD+ (e.g. 2400x1080)"], "refresh_hz": [60, 90],
            "battery_mah": (4000, 5000), "main_camera_mp": [13, 48, 50], "weight_g": (180, 210)
        },
        "Mid-Range": {
            "price": (400, 700), "ram_gb": [6, 8], "storage_gb": [128, 256],
            "screen_inches": (6.4, 6.7), "resolution": ["FHD+ (e.g. 2400x1080)"], "refresh_hz": [90, 120],
            "battery_mah": (4500, 5000), "main_camera_mp": [50, 64, 108], "weight_g": (170, 200)
        },
        "Flagship": {
            "price": (700, 1500), "ram_gb": [8, 12, 16], "storage_gb": [256, 512, 1000],
            "screen_inches": (6.5, 6.9), "resolution": ["FHD+ (e.g. 2400x1080)", "QHD+ (e.g. 3200x1440)"], "refresh_hz": [120], # Often LTPO 1-120Hz
            "battery_mah": (4500, 5500), "main_camera_mp": [50, 108, 200], # Plus secondary cameras
            "weight_g": (170, 230)
        }
    },
    "Monitor": {
        "Office/Budget": {
            "price": (100, 250), "screen_inches": [21.5, 23.8, 27],
            "resolution": ["1920x1080"], "panel_type": ["VA", "IPS"], "refresh_hz": [60, 75],
            "response_time_ms": [4, 5], "brightness_nits": (250, 300), "color_gamut_srgb": (95, 100)
        },
        "Mainstream/Gaming": {
            "price": (250, 600), "screen_inches": [27, 31.5],
            "resolution": ["1920x1080", "2560x1440"], "panel_type": ["IPS", "Fast VA"], "refresh_hz": [144, 165, 180],
            "response_time_ms": [1, 2], "brightness_nits": (300, 400), "color_gamut_dcip3": (90, 98) # Often DCI-P3 for gaming
        },
        "Professional/High-End": {
            "price": (500, 2000), "screen_inches": [27, 32, 34], # Include Ultrawide potential implicitly
            "resolution": ["2560x1440", "3840x2160", "3440x1440"], "panel_type": ["IPS", "OLED"], "refresh_hz": [60, 120, 144],
            "response_time_ms": [1, 4, 5], "brightness_nits": (350, 600), # Check HDR specs too
            "color_gamut_adobergb": (95, 100) # Often AdobeRGB or high DCI-P3 for pro work
        }
    }
}

# --- LLM Client Initialization ---
try:
    llm_call = create_llm_client()
    print("LLM Client created successfully.")
except Exception as e:
    print(f"Error creating LLM client: {e}. Exiting.")
    exit()

# --- LLM Call Wrapper ---
def call_llm_api(prompt_text):
    message = user_struct(prompt_text)
    response = llm_call([message]) 
    return response

# --- Generation Loop ---
generated_items = []
product_types = list(spec_templates.keys())

print(f"Starting generation of {NUM_ITEMS_TO_GENERATE} items...")

for i in range(NUM_ITEMS_TO_GENERATE):
    product_type = random.choice(product_types)
    tiers = list(spec_templates[product_type].keys())
    tier_name = random.choice(tiers)
    template = spec_templates[product_type][tier_name]

    print(f"\n[{i+1}/{NUM_ITEMS_TO_GENERATE}] Generating: {tier_name} {product_type}")

    # Generate specs based on the template
    specs = {}
    spec_details_string = ""
    readable_key_map = { # Map technical keys to more readable names for the prompt
        "ram_gb": "RAM", "storage_gb": "Storage", "cpu": "Processor", "gpu": "Graphics",
        "screen_inches": "Screen Size", "resolution": "Resolution", "refresh_hz": "Refresh Rate",
        "battery_hours": "Battery Life (Hours)", "weight_kg": "Weight (kg)", "price": "Price",
        "battery_mah": "Battery Capacity (mAh)", "main_camera_mp": "Main Camera", "weight_g": "Weight (g)",
        "panel_type": "Panel Type", "response_time_ms": "Response Time (ms)",
        "brightness_nits": "Brightness (nits)", "color_gamut_srgb": "sRGB Coverage (%)",
        "color_gamut_dcip3": "DCI-P3 Coverage (%)", "color_gamut_adobergb": "Adobe RGB Coverage (%)"
    }

    for key, value in template.items():
        generated_value = None
        unit = ""
        if isinstance(value, list):
            generated_value = random.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            if all(isinstance(x, int) for x in value):
                generated_value = random.randint(value[0], value[1])
            elif all(isinstance(x, float) for x in value):
                generated_value = round(random.uniform(value[0], value[1]), 2 if key == 'price' else 1)
        else:
            print(f"Warning: Unsupported template format for key '{key}' in {product_type}/{tier_name}")
            continue
        
        specs[key] = generated_value
        
        # Format for prompt string
        readable_key = readable_key_map.get(key, key.replace('_', ' ').title())
        formatted_value = generated_value
        
        # Add units/context for prompt clarity
        if key == 'price': unit = "$" ; formatted_value = f"{generated_value:,.2f}"
        elif 'ram_gb' == key: unit = " GB"
        elif 'storage_gb' == key: unit = " GB" if generated_value < 1000 else " TB"; formatted_value = generated_value if generated_value < 1000 else generated_value // 1000
        elif 'screen_inches' == key: unit = '"'
        elif 'weight_kg' == key: unit = " kg"
        elif 'weight_g' == key: unit = " g"
        elif 'battery_hours' == key: unit = " hours (est.)"
        elif 'battery_mah' == key: unit = " mAh"
        elif 'main_camera_mp' == key: unit = " MP"
        elif 'refresh_hz' == key: unit = " Hz"
        elif 'response_time_ms' == key: unit = " ms"
        elif 'brightness_nits' == key: unit = " nits"
        elif 'color_gamut' in key: unit = "%"

        spec_details_string += f"- {readable_key}: {formatted_value}{unit}\n"

    # Construct the prompt
    prompt = f"""
Generate a realistic product description for the following electronic item:

Product Type: {product_type}
Tier: {tier_name}
Specifications:
{spec_details_string.strip()}

Instructions:
- Weave ALL the specifications listed above naturally into a compelling product description paragraph or two.
- The tone should be typical for marketing or product information found on retail websites (e.g., engaging but informative).
- Highlight one or two key strengths based on the provided specifications (e.g., portability, performance, display quality, battery life, value). Choose strengths relevant to the tier and product type.
- Ensure the description is coherent, easy to read, and sounds authentic. Avoid just listing the numbers bluntly or starting every sentence the same way.
- Do not invent significant features or specifications not listed above. You can add minor connecting phrases or common features implied by the specs (e.g., mention SSD speed implicitly if storage is high).
"""

    # Call LLM
    print(f"  Calling LLM for {tier_name} {product_type}...")
    description = call_llm_api(prompt)

    if description:
        print(f"  Received description (length: {len(description)} chars).")
        generated_items.append({
            "id": i,
            "product_type": product_type,
            "tier": tier_name,
            "specs": specs, # Store the raw generated specs
            "description": description.strip()
        })
    else:
        print(f"  Failed to get description from LLM for item {i}. Skipping.")

    # Optional delay
    if LLM_REQUEST_DELAY_SECONDS > 0 and i < NUM_ITEMS_TO_GENERATE - 1:
         print(f"  Waiting {LLM_REQUEST_DELAY_SECONDS}s before next request...")
         time.sleep(LLM_REQUEST_DELAY_SECONDS)

# --- Save Results ---
print(f"\nGenerated {len(generated_items)} items successfully.")
output_dir = "generated_data"
os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
output_path = os.path.join(output_dir, OUTPUT_FILENAME)

dumpj(generated_items, OUTPUT_FILENAME)
print("Data saved successfully.")

print("Script finished.")