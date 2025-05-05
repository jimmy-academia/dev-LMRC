#!/usr/bin/env python3
"""
Improved Synthetic Dataset Generator for Testing Retrieval Methods

This script generates:
1. Synthetic product data with realistic specifications
2. Corresponding search queries based on hard and soft constraints
3. Ground truth relevance rankings with proper constraint filtering and weighted scoring

The output is saved in formats compatible with baseline retrieval methods.
"""

import os
import json
import random
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from utils import dumpj, create_llm_client, user_struct, system_struct, ensure_dir

# --- Configuration ---
NUM_PRODUCTS = 200               # Total number of products to generate
NUM_REQUESTS = 50                # Total number of requests to generate
OUTPUT_DIR = "generated_data"    # Output directory
LLM_DELAY = 1                    # Delay between LLM calls to avoid rate limits
PRODUCT_FILENAME = "products.json"
REQUESTS_FILENAME = "requests.json"
COMPATIBLE_FILENAME = "baseline_compatible.json"

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

# --- Query Types for Request Generation ---
QUERY_TYPES = [
    {
        "name": "specific_need",
        "description": "User has a specific need with both hard and soft requirements",
        "example": "Need a laptop under $1000 for graphic design work",
        "hard_constraints": 1,
        "soft_constraints": 2
    },
    {
        "name": "feature_focused",
        "description": "User specifies must-have features and preferences",
        "example": "Gaming monitor with at least 144Hz and good color accuracy",
        "hard_constraints": 1,
        "soft_constraints": 1
    },
    {
        "name": "budget_constrained",
        "description": "User has strict budget with other preferences",
        "example": "Smartphone below $500 with good camera",
        "hard_constraints": 1, 
        "soft_constraints": 2
    },
    {
        "name": "multi_constraint",
        "description": "Multiple hard requirements and preferences",
        "example": "Laptop with 16GB RAM minimum and under $1200, lightweight and long battery",
        "hard_constraints": 2,
        "soft_constraints": 2
    },
    {
        "name": "performance_focused",
        "description": "Minimum performance requirements and preferences",
        "example": "Monitor with at least 120Hz and 1440p, good for design work",
        "hard_constraints": 2,
        "soft_constraints": 1
    }
]

# --- Constraint Generation Utilities ---

def select_constraint_specs(product_type, tier, specs, num_hard, num_soft):
    """Select specifications to use as constraints based on product type."""
    # Define key specs that are most commonly used as constraints by product type
    key_specs_by_type = {
        "Laptop": {
            "hard_candidates": ["price", "ram_gb", "storage_gb", "screen_inches", "gpu", "cpu"],
            "soft_candidates": ["battery_hours", "weight_kg", "resolution", "refresh_hz"],
            "ranges": {
                "price": (300, 4000),
                "ram_gb": (4, 64),
                "storage_gb": (128, 4000),
                "screen_inches": (11, 18),
                "refresh_hz": (60, 360),
                "battery_hours": (2, 18),
                "weight_kg": (0.8, 3.5)
            }
        },
        "Smartphone": {
            "hard_candidates": ["price", "ram_gb", "storage_gb", "main_camera_mp"],
            "soft_candidates": ["battery_mah", "screen_inches", "refresh_hz", "weight_g"],
            "ranges": {
                "price": (100, 2000),
                "ram_gb": (3, 20),
                "storage_gb": (32, 1000),
                "main_camera_mp": (8, 200),
                "battery_mah": (3000, 6000),
                "refresh_hz": (60, 120),
                "weight_g": (120, 250)
            }
        },
        "Monitor": {
            "hard_candidates": ["price", "screen_inches", "resolution", "refresh_hz", "panel_type"],
            "soft_candidates": ["response_time_ms", "brightness_nits", "color_gamut_srgb", "color_gamut_dcip3", "color_gamut_adobergb"],
            "ranges": {
                "price": (80, 2500),
                "screen_inches": (21, 49),
                "refresh_hz": (60, 360),
                "response_time_ms": (0.1, 5),
                "brightness_nits": (200, 1000)
            }
        }
    }
    
    candidates = key_specs_by_type.get(product_type, {
        "hard_candidates": list(specs.keys())[:3],
        "soft_candidates": list(specs.keys())[3:],
        "ranges": {}
    })
    
    # Select hard constraints
    hard_specs = random.sample(candidates["hard_candidates"], min(num_hard, len(candidates["hard_candidates"])))
    
    # Select soft constraints (different from hard constraints)
    available_soft = [spec for spec in candidates["soft_candidates"] if spec not in hard_specs]
    if len(available_soft) < num_soft:
        # If we need more soft constraints, include some hard candidates that weren't selected
        available_soft += [spec for spec in candidates["hard_candidates"] if spec not in hard_specs]
    
    soft_specs = random.sample(available_soft, min(num_soft, len(available_soft)))
    
    # Generate constraint values
    hard_constraints = []
    soft_constraints = []
    ranges = candidates["ranges"]
    
    for spec in hard_specs:
        constraint = generate_constraint_value(spec, specs.get(spec), ranges.get(spec), "hard", product_type)
        if constraint:
            hard_constraints.append(constraint)
    
    for spec in soft_specs:
        constraint = generate_constraint_value(spec, specs.get(spec), ranges.get(spec), "soft", product_type)
        if constraint:
            # Assign a random weight (1-5) to each soft constraint
            constraint["weight"] = random.randint(1, 5)
            soft_constraints.append(constraint)
    
    return hard_constraints, soft_constraints

def generate_constraint_value(spec_name, current_value, range_info, constraint_type, product_type):
    """Generate a constraint value based on the spec type and current value."""
    if current_value is None:
        return None
    
    # For the target product, we need to generate constraints that it will satisfy
    constraint = {
        "spec": spec_name,
        "type": constraint_type
    }
    
    # Handle different types of specs differently
    if isinstance(current_value, (int, float)):
        # For numeric specs, generate minimum/maximum constraints
        if spec_name == "price":
            # For price, usually set a maximum
            if constraint_type == "hard":
                # Hard constraint: set maximum price slightly above current
                max_price = current_value * random.uniform(1.0, 1.2)
                constraint["operator"] = "max"
                constraint["value"] = round(max_price, 2)
            else:
                # Soft constraint: prefer lower prices
                constraint["operator"] = "prefer_lower"
                constraint["value"] = "lower"
        
        elif spec_name in ["ram_gb", "storage_gb", "main_camera_mp", "refresh_hz", "battery_hours", "battery_mah"]:
            # For these specs, usually set a minimum
            if constraint_type == "hard":
                # Hard constraint: set minimum requirement at or below current value
                min_value = current_value * random.uniform(0.8, 1.0)
                constraint["operator"] = "min"
                constraint["value"] = int(min_value) if isinstance(current_value, int) else round(min_value, 1)
            else:
                # Soft constraint: prefer higher values
                constraint["operator"] = "prefer_higher"
                constraint["value"] = "higher"
        
        elif spec_name in ["weight_kg", "weight_g", "response_time_ms"]:
            # For these specs, lower is usually better
            if constraint_type == "hard":
                # Hard constraint: set maximum at or above current value
                max_value = current_value * random.uniform(1.0, 1.3)
                constraint["operator"] = "max"
                constraint["value"] = int(max_value) if isinstance(current_value, int) else round(max_value, 1)
            else:
                # Soft constraint: prefer lower values
                constraint["operator"] = "prefer_lower"
                constraint["value"] = "lower"
        
        elif spec_name in ["screen_inches", "brightness_nits"]:
            # These can go either way depending on preference
            if random.random() < 0.5:
                if constraint_type == "hard":
                    # Minimum size/brightness
                    min_value = current_value * random.uniform(0.8, 1.0)
                    constraint["operator"] = "min"
                    constraint["value"] = round(min_value, 1)
                else:
                    # Prefer larger/brighter
                    constraint["operator"] = "prefer_higher"
                    constraint["value"] = "higher"
            else:
                if constraint_type == "hard":
                    # Maximum size/brightness
                    max_value = current_value * random.uniform(1.0, 1.3)
                    constraint["operator"] = "max"
                    constraint["value"] = round(max_value, 1)
                else:
                    # Prefer smaller/dimmer
                    constraint["operator"] = "prefer_lower"
                    constraint["value"] = "lower"
    
    elif isinstance(current_value, str):
        # For string specs like resolution, panel_type, etc.
        if spec_name == "resolution":
            # Resolution can be specified as a minimum
            if constraint_type == "hard":
                constraint["operator"] = "min"
                constraint["value"] = current_value
            else:
                constraint["operator"] = "prefer"
                constraint["value"] = current_value
        
        elif spec_name in ["panel_type", "cpu", "gpu"]:
            if constraint_type == "hard":
                constraint["operator"] = "is"
                constraint["value"] = current_value
            else:
                constraint["operator"] = "prefer"
                constraint["value"] = current_value
    
    # Add human-readable description
    constraint["description"] = format_constraint_description(constraint)
    
    return constraint

def format_constraint_description(constraint):
    """Format a constraint into a human-readable description."""
    spec = constraint["spec"]
    operator = constraint["operator"]
    value = constraint["value"]
    
    # Map spec names to more readable forms
    readable_specs = {
        "price": "price",
        "ram_gb": "RAM",
        "storage_gb": "storage",
        "screen_inches": "screen size",
        "refresh_hz": "refresh rate",
        "battery_hours": "battery life",
        "weight_kg": "weight",
        "resolution": "resolution",
        "main_camera_mp": "camera",
        "battery_mah": "battery capacity",
        "weight_g": "weight",
        "panel_type": "panel type",
        "response_time_ms": "response time",
        "brightness_nits": "brightness",
        "cpu": "processor",
        "gpu": "graphics"
    }
    
    spec_readable = readable_specs.get(spec, spec)
    
    # Format based on operator type
    if operator == "min":
        # Add appropriate units
        unit = ""
        if spec == "ram_gb":
            unit = " GB"
            if isinstance(value, (int, float)) and value >= 1000:
                value = value / 1000
                unit = " TB"
        elif spec == "storage_gb":
            if isinstance(value, (int, float)) and value >= 1000:
                value = value / 1000
                unit = " TB"
            else:
                unit = " GB"
        elif spec == "refresh_hz":
            unit = " Hz"
        elif spec == "battery_hours":
            unit = " hours"
        elif spec == "screen_inches":
            unit = "\""
        elif spec == "weight_kg":
            unit = " kg"
        elif spec == "weight_g":
            unit = " g"
        elif spec == "battery_mah":
            unit = " mAh"
        elif spec == "response_time_ms":
            unit = " ms"
        elif spec == "brightness_nits":
            unit = " nits"
        elif spec == "main_camera_mp":
            unit = " MP"
            
        return f"Minimum {spec_readable} of {value}{unit}"
    
    elif operator == "max":
        # Add appropriate units
        unit = ""
        if spec == "price":
            return f"Maximum price of ${value:,.2f}"
        elif spec == "ram_gb":
            unit = " GB"
        elif spec == "storage_gb":
            if isinstance(value, (int, float)) and value >= 1000:
                value = value / 1000
                unit = " TB"
            else:
                unit = " GB"
        elif spec == "refresh_hz":
            unit = " Hz"
        elif spec == "battery_hours":
            unit = " hours"
        elif spec == "screen_inches":
            unit = "\""
        elif spec == "weight_kg":
            unit = " kg"
        elif spec == "weight_g":
            unit = " g"
        elif spec == "response_time_ms":
            unit = " ms"
        elif spec == "brightness_nits":
            unit = " nits"
            
        return f"Maximum {spec_readable} of {value}{unit}"
    
    elif operator == "is":
        return f"{spec_readable} is {value}"
    
    elif operator == "prefer_higher":
        return f"Prefer higher {spec_readable}"
    
    elif operator == "prefer_lower":
        return f"Prefer lower {spec_readable}"
    
    elif operator == "prefer":
        return f"Prefer {value} {spec_readable}"
    
    return f"{spec_readable} {operator} {value}"

def generate_products(llm_client, num_products=NUM_PRODUCTS):
    """Generate synthetic product data with realistic specifications."""
    generated_products = []
    product_types = list(spec_templates.keys())
    
    print(f"Generating {num_products} synthetic products...")
    
    for i in range(num_products):
        product_type = random.choice(product_types)
        tiers = list(spec_templates[product_type].keys())
        tier_name = random.choice(tiers)
        template = spec_templates[product_type][tier_name]
        
        print(f"\n[{i+1}/{num_products}] Generating: {tier_name} {product_type}")
        
        # Generate specs based on the template
        specs = {}
        spec_details_string = ""
        readable_key_map = {
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
            elif 'storage_gb' == key: 
                unit = " GB" if generated_value < 1000 else " TB"
                formatted_value = generated_value if generated_value < 1000 else generated_value // 1000
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
        print(f"  Calling LLM for description...")
        description = llm_client([user_struct(prompt)])
        
        if description:
            print(f"  Received description ({len(description)} chars)")
            product = {
                "id": i,
                "product_type": product_type,
                "tier": tier_name,
                "specs": specs,
                "description": description.strip()
            }
            generated_products.append(product)
        else:
            print(f"  Failed to get description for item {i}. Skipping.")
        
        # Delay between requests
        if LLM_DELAY > 0 and i < num_products - 1:
            time.sleep(LLM_DELAY)
    
    return generated_products

def generate_request_query(llm_client, target_product, hard_constraints, soft_constraints, query_type):
    """Generate a search query based on the specified constraints."""
    product_type = target_product["product_type"]
    tier = target_product["tier"]
    
    # Format constraints for the prompt
    hard_constraints_text = "\n".join([f"- {c['description']}" for c in hard_constraints])
    
    # Add weights to soft constraint descriptions
    soft_constraints_with_weights = []
    for c in soft_constraints:
        weight = c.get("weight", 3)  # Default to medium weight (3/5)
        weight_text = "High" if weight >= 4 else "Medium" if weight >= 2 else "Low"
        soft_constraints_with_weights.append(f"- {c['description']} (Importance: {weight_text})")
    
    soft_constraints_text = "\n".join(soft_constraints_with_weights)
    
    # Construct the prompt
    prompt = f"""
Generate a realistic search query for a {product_type} with the following requirements.

Product Type: {product_type}
Query Type: {query_type["name"]} - {query_type["description"]}

Hard Requirements (must be satisfied):
{hard_constraints_text}

Preferences (desirable but optional, with varying importance):
{soft_constraints_text}

Instructions:
1. Create a natural search query (15-30 words) that someone might type when looking for a product with these requirements
2. Make the hard requirements clear (using words like "must have", "need", "required", etc.)
3. Express the preferences with language that indicates their importance (based on the importance level)
4. Phrase the query as a natural request, not a bullet list of requirements
5. Do not mention brand names or exact model numbers
6. The query should sound like something a real person would type into a search box

Return ONLY the search query text, with no additional explanation or formatting.
"""
    
    # Call LLM
    response = llm_client([
        system_struct("You generate realistic product search queries based on specifications and requirements."),
        user_struct(prompt)
    ])
    
    return response.strip()

def check_hard_constraints(product, constraints):
    """
    Check if a product satisfies all hard constraints.
    Returns tuple: (meets_all, results_by_constraint)
    """
    if not constraints:
        return True, {}
    
    results = {}
    for constraint in constraints:
        spec = constraint["spec"]
        operator = constraint["operator"]
        constraint_value = constraint["value"]
        product_value = product["specs"].get(spec)
        
        # Skip if product doesn't have this spec
        if product_value is None:
            results[spec] = False
            continue
        
        # Check constraint based on operator
        if operator == "min":
            results[spec] = product_value >= constraint_value
        elif operator == "max":
            results[spec] = product_value <= constraint_value
        elif operator == "is":
            results[spec] = product_value == constraint_value
        else:
            # Unknown operator, assume success
            results[spec] = True
    
    # All constraints must be satisfied (AND operation)
    meets_all = all(results.values())
    
    return meets_all, results

def calculate_soft_constraint_score(product, constraints):
    """
    Calculate a weighted score based on soft constraints.
    Returns tuple: (overall_score, scores_by_constraint)
    """
    if not constraints:
        return 0.5, {}  # Middle score if no constraints
    
    scores = {}
    total_weight = 0
    weighted_sum = 0
    
    for constraint in constraints:
        spec = constraint["spec"]
        operator = constraint["operator"]
        constraint_value = constraint["value"]
        weight = constraint.get("weight", 3)  # Default to medium weight
        product_value = product["specs"].get(spec)
        
        # Skip if product doesn't have this spec
        if product_value is None:
            scores[spec] = 0.0
            continue
        
        # Calculate score based on operator
        score = 0.5  # Default to middle score
        
        if operator == "prefer_higher":
            # Need to normalize based on range for the spec
            if spec == "ram_gb":
                score = normalize_value(product_value, 4, 64)
            elif spec == "storage_gb":
                score = normalize_value(product_value, 64, 4000)
            elif spec == "screen_inches":
                score = normalize_value(product_value, 13, 18)
            elif spec == "refresh_hz":
                score = normalize_value(product_value, 60, 360)
            elif spec == "battery_hours":
                score = normalize_value(product_value, 3, 18)
            elif spec == "battery_mah":
                score = normalize_value(product_value, 3000, 6000)
            elif spec == "main_camera_mp":
                score = normalize_value(product_value, 8, 200)
            elif spec == "brightness_nits":
                score = normalize_value(product_value, 250, 1000)
            else:
                # For other specs, use a default range
                score = 0.7 if product_value > average_value_for_spec(spec) else 0.3
        
        elif operator == "prefer_lower":
            # Reverse normalization (lower is better)
            if spec == "price":
                score = 1.0 - normalize_value(product_value, 100, 3500)
            elif spec == "weight_kg":
                score = 1.0 - normalize_value(product_value, 0.8, 3.5)
            elif spec == "weight_g":
                score = 1.0 - normalize_value(product_value, 120, 250)
            elif spec == "response_time_ms":
                score = 1.0 - normalize_value(product_value, 0.1, 5.0)
            else:
                # For other specs, use a default range
                score = 0.7 if product_value < average_value_for_spec(spec) else 0.3
        
        elif operator == "prefer":
            # For exact matches (resolution, panel type, etc.)
            if product_value == constraint_value:
                score = 1.0
            else:
                score = 0.2
        
        # Store the score and add to weighted sum
        scores[spec] = score
        weighted_sum += score * weight
        total_weight += weight
    
    # Calculate weighted average
    if total_weight > 0:
        overall_score = weighted_sum / total_weight
    else:
        overall_score = 0.5
    
    return overall_score, scores

def normalize_value(value, min_val, max_val):
    """Normalize a value to a 0-1 scale based on min/max range."""
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)

def average_value_for_spec(spec):
    """Return a reasonable average value for a spec type."""
    averages = {
        "price": 800,
        "ram_gb": 16,
        "storage_gb": 512,
        "screen_inches": 15,
        "refresh_hz": 120,
        "battery_hours": 8,
        "weight_kg": 1.5,
        "weight_g": 180,
        "battery_mah": 4500,
        "main_camera_mp": 50,
        "response_time_ms": 2,
        "brightness_nits": 350
    }
    return averages.get(spec, 0)

def generate_relevance_scores(products, target_product, hard_constraints, soft_constraints):
    """Generate ground truth relevance scores for all products based on constraints."""
    relevance_scores = []
    
    for product in products:
        # Check hard constraints first (AND operation)
        meets_hard, hard_results = check_hard_constraints(product, hard_constraints)
        
        # Calculate soft constraint score only if hard constraints are met
        if meets_hard:
            soft_score, soft_results = calculate_soft_constraint_score(product, soft_constraints)
        else:
            soft_score, soft_results = 0.0, {}
        
        # The overall score is 0 if hard constraints not met
        overall_score = soft_score if meets_hard else 0.0
        
        # Boost score for target product (to ensure it ranks well)
        if product["id"] == target_product["id"]:
            overall_score = min(1.0, overall_score * 1.1)
        
        relevance_scores.append({
            "product_id": product["id"],
            "product_type": product["product_type"],
            "tier": product["tier"],
            "meets_hard_constraints": meets_hard,
            "hard_constraint_results": hard_results,
            "soft_constraint_score": soft_score,
            "soft_constraint_results": soft_results,
            "overall_score": overall_score
        })
    
    # Sort by overall score (descending)
    relevance_scores.sort(key=lambda x: x["overall_score"], reverse=True)
    
    return relevance_scores

def generate_requests(llm_client, products, num_requests=NUM_REQUESTS):
    """Generate search queries with hard and soft constraints."""
    generated_requests = []
    
    # Select target products randomly
    target_indices = random.sample(range(len(products)), min(num_requests, len(products)))
    
    print(f"\nGenerating {len(target_indices)} search queries...")
    
    for i, idx in enumerate(tqdm(target_indices, desc="Generating queries")):
        target_product = products[idx]
        query_type = random.choice(QUERY_TYPES)
        
        print(f"\n[{i+1}/{len(target_indices)}] Generating {query_type['name']} query for {target_product['product_type']} (Tier: {target_product['tier']})")
        
        # Step 1: Generate constraint specifications
        hard_constraints, soft_constraints = select_constraint_specs(
            target_product["product_type"],
            target_product["tier"],
            target_product["specs"],
            query_type["hard_constraints"],
            query_type["soft_constraints"]
        )
        
        # Print constraints
        print("  Hard Constraints:")
        for constraint in hard_constraints:
            print(f"  - {constraint['description']}")
        
        print("  Soft Constraints:")
        for constraint in soft_constraints:
            print(f"  - {constraint['description']} (Weight: {constraint['weight']})")
        
        # Step 2: Generate search query based on constraints
        query_text = generate_request_query(
            llm_client, 
            target_product, 
            hard_constraints, 
            soft_constraints, 
            query_type
        )
        
        if not query_text:
            print(f"  Failed to generate query for product {idx}. Skipping.")
            continue
            
        print(f"  Query: {query_text}")
        
        # Step 3: Generate relevance scores
        relevance_scores = generate_relevance_scores(
            products, 
            target_product, 
            hard_constraints, 
            soft_constraints
        )
        
        # Create request
        request = {
            "id": len(generated_requests),
            "query": query_text,
            "target_product_id": target_product["id"],
            "product_type": target_product["product_type"],
            "tier": target_product["tier"],
            "query_type": query_type["name"],
            "hard_constraints": [c for c in hard_constraints],
            "soft_constraints": [c for c in soft_constraints],
            "relevance_scores": relevance_scores[:20]  # Top 20 most relevant products
        }
        
        generated_requests.append(request)
        
        # Delay between requests
        if LLM_DELAY > 0 and i < len(target_indices) - 1:
            time.sleep(LLM_DELAY)
    
    return generated_requests

def create_baseline_compatible_format(products, requests):
    """Create a format compatible with baseline retrieval methods."""
    compatible_items = []
    for product in products:
        compatible_items.append({
            "item_id": str(product["id"]),
            "metadata": product["description"],
            "summary": f"{product['product_type']} - {product['tier']}",
            "category": product["product_type"]
        })
    
    compatible_requests = []
    for req in requests:
        compatible_requests.append({
            "item_id": str(req["target_product_id"]),
            "query": req["query"],
            "qid": str(req["id"]),
            "user_id": "synthetic_user"
        })
    
    return {
        "compatible_items": compatible_items,
        "compatible_requests": compatible_requests
    }

def main():
    # Create output directory
    ensure_dir(OUTPUT_DIR)
    
    # Initialize LLM client
    llm_client = create_llm_client()
    print("LLM Client initialized successfully.")
    
    product_path = os.path.join(OUTPUT_DIR, PRODUCT_FILENAME)
    
    # Check if products already exist
    if os.path.exists(product_path):
        print(f"Loading existing products from {product_path}")
        with open(product_path, 'r') as f:
            products = json.load(f)
    else:
        # Generate products
        products = generate_products(llm_client, NUM_PRODUCTS)
        
        # Save products
        dumpj(products, product_path)
        print(f"Generated and saved {len(products)} products to {product_path}")
    
    # Generate requests
    requests = generate_requests(llm_client, products, NUM_REQUESTS)
    
    # Save requests
    requests_path = os.path.join(OUTPUT_DIR, REQUESTS_FILENAME)
    dumpj(requests, requests_path)
    print(f"Generated and saved {len(requests)} requests to {requests_path}")
    
    # Create baseline compatible format
    compatible_data = create_baseline_compatible_format(products, requests)
    
    # Save compatible data
    compatible_path = os.path.join(OUTPUT_DIR, COMPATIBLE_FILENAME)
    dumpj(compatible_data, compatible_path)
    print(f"Created and saved baseline-compatible format to {compatible_path}")
    
    # Print summary
    print("\nDataset Generation Complete")
    print(f"Products: {len(products)}")
    print(f"Requests: {len(requests)}")
    print(f"All files saved to directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()