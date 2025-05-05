#!/usr/bin/env python3
# run => python -m data.synth
"""
Streamlined Synthetic Dataset Generator for Testing Retrieval Methods

Generates realistic product data, search queries with constraints, and relevance rankings.
"""

import random
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from utils import dumpj, create_llm_client, user_struct, system_struct, ensure_dir

# --- Configuration ---
CONFIG = {
    "num_products": 200,
    "num_requests": 50,
    "output_dir": "cache/synth_products",
    "llm_delay": 0,
    "filenames": {
        "products": "products.json",
        "requests": "requests.json"
    }
}

# --- Product Templates ---
PRODUCT_TEMPLATES = {
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
            "battery_hours": (3, 6), "weight_kg": (1.8, 3.0)
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
            "screen_inches": (6.5, 6.9), "resolution": ["FHD+ (e.g. 2400x1080)", "QHD+ (e.g. 3200x1440)"], "refresh_hz": [120],
            "battery_mah": (4500, 5500), "main_camera_mp": [50, 108, 200],
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
            "response_time_ms": [1, 2], "brightness_nits": (300, 400), "color_gamut_dcip3": (90, 98)
        },
        "Professional/High-End": {
            "price": (500, 2000), "screen_inches": [27, 32, 34],
            "resolution": ["2560x1440", "3840x2160", "3440x1440"], "panel_type": ["IPS", "OLED"], "refresh_hz": [60, 120, 144],
            "response_time_ms": [1, 4, 5], "brightness_nits": (350, 600),
            "color_gamut_adobergb": (95, 100)
        }
    }
}

# --- Query Types ---
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

# --- Readable Key Mapping ---
READABLE_KEY_MAP = {
    "ram_gb": "RAM", "storage_gb": "Storage", "cpu": "Processor", "gpu": "Graphics",
    "screen_inches": "Screen Size", "resolution": "Resolution", "refresh_hz": "Refresh Rate",
    "battery_hours": "Battery Life (Hours)", "weight_kg": "Weight (kg)", "price": "Price",
    "battery_mah": "Battery Capacity (mAh)", "main_camera_mp": "Main Camera", "weight_g": "Weight (g)",
    "panel_type": "Panel Type", "response_time_ms": "Response Time (ms)",
    "brightness_nits": "Brightness (nits)", "color_gamut_srgb": "sRGB Coverage (%)",
    "color_gamut_dcip3": "DCI-P3 Coverage (%)", "color_gamut_adobergb": "Adobe RGB Coverage (%)"
}

# --- Spec Ranges for Normalization ---
SPEC_RANGES = {
    "price": (80, 4000),
    "ram_gb": (4, 64),
    "storage_gb": (64, 4000),
    "screen_inches": (6, 34),  # Combines both phone and monitor ranges
    "refresh_hz": (60, 360),
    "battery_hours": (2, 18),
    "weight_kg": (0.8, 3.5),
    "battery_mah": (3000, 6000),
    "main_camera_mp": (8, 200),
    "weight_g": (120, 250),
    "response_time_ms": (0.1, 5),
    "brightness_nits": (200, 1000)
}

# --- Average Values for Specs ---
AVG_SPEC_VALUES = {
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

def generate_product_specs(product_type, tier_name):
    """Generate product specifications based on template."""
    template = PRODUCT_TEMPLATES[product_type][tier_name]
    specs = {}
    spec_details = []
    
    for key, value in template.items():
        if isinstance(value, list):
            generated_value = random.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            if all(isinstance(x, int) for x in value):
                generated_value = random.randint(value[0], value[1])
            elif all(isinstance(x, float) for x in value):
                generated_value = round(random.uniform(value[0], value[1]), 2 if key == 'price' else 1)
        else:
            continue
        
        specs[key] = generated_value
        
        # Format for prompt string
        readable_key = READABLE_KEY_MAP.get(key, key.replace('_', ' ').title())
        formatted_value = generated_value
        
        # Add units/context for prompt clarity
        unit = ""
        if key == 'price': 
            unit = "$" 
            formatted_value = f"{generated_value:,.2f}"
        elif key == 'ram_gb': 
            unit = " GB"
        elif key == 'storage_gb': 
            unit = " GB" if generated_value < 1000 else " TB"
            formatted_value = generated_value if generated_value < 1000 else generated_value // 1000
        elif key == 'screen_inches': 
            unit = '"'
        elif key == 'weight_kg': 
            unit = " kg"
        elif key == 'weight_g': 
            unit = " g"
        elif key == 'battery_hours': 
            unit = " hours (est.)"
        elif key == 'battery_mah': 
            unit = " mAh"
        elif key == 'main_camera_mp': 
            unit = " MP"
        elif key == 'refresh_hz': 
            unit = " Hz"
        elif key == 'response_time_ms': 
            unit = " ms"
        elif key == 'brightness_nits': 
            unit = " nits"
        elif 'color_gamut' in key: 
            unit = "%"
        
        spec_details.append(f"- {readable_key}: {formatted_value}{unit}")
    
    return specs, "\n".join(spec_details)

def generate_product_description(llm_client, product_type, tier_name, spec_details):
    """Generate product description using LLM."""
    prompt = f"""
Generate a realistic product description for the following electronic item:

Product Type: {product_type}
Tier: {tier_name}
Specifications:
{spec_details}

Instructions:
- Weave ALL the specifications into a compelling product description paragraph.
- Use engaging but informative tone typical for retail websites.
- Highlight 1-2 key strengths based on the specifications.
- Make the description coherent, easy to read, and authentic.
- Don't invent significant features not listed above.
"""
    
    return llm_client([user_struct(prompt)])

def generate_products(llm_client, num_products=CONFIG["num_products"]):
    """Generate synthetic product data with realistic specifications."""
    products = []
    product_types = list(PRODUCT_TEMPLATES.keys())
    
    print(f"Generating {num_products} synthetic products...")
    
    for i in range(num_products):
        product_type = random.choice(product_types)
        tiers = list(PRODUCT_TEMPLATES[product_type].keys())
        tier_name = random.choice(tiers)
        
        print(f"\n[{i+1}/{num_products}] Generating: {tier_name} {product_type}")
        
        # Generate specs and format for prompt
        specs, spec_details = generate_product_specs(product_type, tier_name)
        
        # Generate description
        print(f"  Calling LLM for description...")
        description = generate_product_description(llm_client, product_type, tier_name, spec_details)
        
        if description:
            print(f"  Received description ({len(description)} chars)")
            products.append({
                "id": i,
                "product_type": product_type,
                "tier": tier_name,
                "specs": specs,
                "description": description.strip()
            })
        else:
            print(f"  Failed to get description for item {i}. Skipping.")
        
        # Delay between requests
        if i < num_products - 1 and CONFIG["llm_delay"] > 0:
            time.sleep(CONFIG["llm_delay"])
        
    return products

def select_constraint_key_candidates(product_type):
    """Get constraint key candidates for a product type."""
    key_specs_by_type = {
        "Laptop": {
            "hard_candidates": ["price", "ram_gb", "storage_gb", "screen_inches", "gpu", "cpu"],
            "soft_candidates": ["battery_hours", "weight_kg", "resolution", "refresh_hz"]
        },
        "Smartphone": {
            "hard_candidates": ["price", "ram_gb", "storage_gb", "main_camera_mp"],
            "soft_candidates": ["battery_mah", "screen_inches", "refresh_hz", "weight_g"]
        },
        "Monitor": {
            "hard_candidates": ["price", "screen_inches", "resolution", "refresh_hz", "panel_type"],
            "soft_candidates": ["response_time_ms", "brightness_nits", "color_gamut_srgb", "color_gamut_dcip3", "color_gamut_adobergb"]
        }
    }
    
    return key_specs_by_type.get(product_type, {
        "hard_candidates": ["price", "ram_gb", "storage_gb"],
        "soft_candidates": ["screen_inches", "refresh_hz"]
    })

def format_constraint_description(constraint):
    """Format a constraint into a human-readable description."""
    spec = constraint["spec"]
    operator = constraint["operator"]
    value = constraint["value"]
    
    spec_readable = READABLE_KEY_MAP.get(spec, spec.replace('_', ' ').title())
    
    # Format based on operator type
    if operator == "min":
        unit = get_unit_for_spec(spec, value)
        return f"Minimum {spec_readable} of {value}{unit}"
    
    elif operator == "max":
        if spec == "price":
            return f"Maximum price of ${value:,.2f}"
        unit = get_unit_for_spec(spec, value)
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

def get_unit_for_spec(spec, value):
    """Get the appropriate unit for a spec value."""
    if spec == "ram_gb":
        return " GB"
    elif spec == "storage_gb":
        return " TB" if isinstance(value, (int, float)) and value >= 1000 else " GB"
    elif spec == "refresh_hz":
        return " Hz"
    elif spec == "battery_hours":
        return " hours"
    elif spec == "screen_inches":
        return "\""
    elif spec == "weight_kg":
        return " kg"
    elif spec == "weight_g":
        return " g"
    elif spec == "battery_mah":
        return " mAh"
    elif spec == "response_time_ms":
        return " ms"
    elif spec == "brightness_nits":
        return " nits"
    elif spec == "main_camera_mp":
        return " MP"
    return ""

def generate_constraint_value(spec, current_value, constraint_type):
    """Generate a constraint value based on the specification."""
    if current_value is None:
        return None
    
    constraint = {
        "spec": spec,
        "type": constraint_type,
        "operator": "prefer"  # Default operator to ensure it's always defined
    }
    
    # Handle different types of specs
    if isinstance(current_value, (int, float)):
        if spec == "price":
            # For price, set maximum
            if constraint_type == "hard":
                max_price = current_value * random.uniform(1.0, 1.2)
                constraint["operator"] = "max"
                constraint["value"] = round(max_price, 2)
            else:
                constraint["operator"] = "prefer_lower"
                constraint["value"] = "lower"
        
        elif spec in ["ram_gb", "storage_gb", "main_camera_mp", "refresh_hz", "battery_hours", "battery_mah"]:
            # For these specs, set minimum
            if constraint_type == "hard":
                min_value = current_value * random.uniform(0.8, 1.0)
                constraint["operator"] = "min"
                constraint["value"] = int(min_value) if isinstance(current_value, int) else round(min_value, 1)
            else:
                constraint["operator"] = "prefer_higher"
                constraint["value"] = "higher"
        
        elif spec in ["weight_kg", "weight_g", "response_time_ms"]:
            # For these specs, lower is better
            if constraint_type == "hard":
                max_value = current_value * random.uniform(1.0, 1.3)
                constraint["operator"] = "max"
                constraint["value"] = int(max_value) if isinstance(current_value, int) else round(max_value, 1)
            else:
                constraint["operator"] = "prefer_lower"
                constraint["value"] = "lower"
        
        elif spec in ["screen_inches", "brightness_nits"]:
            # These can go either way
            if random.random() < 0.5:
                if constraint_type == "hard":
                    min_value = current_value * random.uniform(0.8, 1.0)
                    constraint["operator"] = "min"
                    constraint["value"] = round(min_value, 1)
                else:
                    constraint["operator"] = "prefer_higher"
                    constraint["value"] = "higher"
            else:
                if constraint_type == "hard":
                    max_value = current_value * random.uniform(1.0, 1.3)
                    constraint["operator"] = "max"
                    constraint["value"] = round(max_value, 1)
                else:
                    constraint["operator"] = "prefer_lower"
                    constraint["value"] = "lower"
    
    elif isinstance(current_value, str):
        # For string specs
        if constraint_type == "hard":
            constraint["operator"] = "is" if spec in ["panel_type", "cpu", "gpu"] else "min"
            constraint["value"] = current_value
        else:
            constraint["operator"] = "prefer"
            constraint["value"] = current_value
    
    # Ensure value is set
    if "value" not in constraint:
        constraint["value"] = current_value
    
    # Add human-readable description
    constraint["description"] = format_constraint_description(constraint)
    
    return constraint

def select_constraints(target_product, num_hard, num_soft):
    """Select constraints for a product based on query requirements."""
    product_type = target_product["product_type"]
    specs = target_product["specs"]
    
    # Get constraint candidates
    candidates = select_constraint_key_candidates(product_type)
    
    # Select hard constraints
    hard_specs = random.sample(candidates["hard_candidates"], min(num_hard, len(candidates["hard_candidates"])))
    
    # Select soft constraints (different from hard constraints)
    available_soft = [spec for spec in candidates["soft_candidates"] if spec not in hard_specs]
    if len(available_soft) < num_soft:
        available_soft += [spec for spec in candidates["hard_candidates"] if spec not in hard_specs]
    
    soft_specs = random.sample(available_soft, min(num_soft, len(available_soft)))
    
    # Generate constraint values
    hard_constraints = []
    soft_constraints = []
    
    for spec in hard_specs:
        constraint = generate_constraint_value(spec, specs.get(spec), "hard")
        if constraint:
            hard_constraints.append(constraint)
    
    for spec in soft_specs:
        constraint = generate_constraint_value(spec, specs.get(spec), "soft")
        if constraint:
            constraint["weight"] = random.randint(1, 5)  # Assign random weight 1-5
            soft_constraints.append(constraint)
    
    return hard_constraints, soft_constraints

def generate_query(llm_client, target_product, hard_constraints, soft_constraints, query_type):
    """Generate a search query based on constraints."""
    product_type = target_product["product_type"]
    
    # Format constraints for the prompt
    hard_constraints_text = "\n".join([f"- {c['description']}" for c in hard_constraints])
    
    # Add weights to soft constraint descriptions
    soft_constraints_with_weights = []
    for c in soft_constraints:
        weight = c.get("weight", 3)
        weight_text = "High" if weight >= 4 else "Medium" if weight >= 2 else "Low"
        soft_constraints_with_weights.append(f"- {c['description']} (Importance: {weight_text})")
    
    soft_constraints_text = "\n".join(soft_constraints_with_weights)
    
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
    
    return llm_client([
        system_struct("You generate realistic product search queries based on specifications and requirements."),
        user_struct(prompt)
    ])

def normalize_value(value, spec):
    """Normalize a value to a 0-1 scale based on min/max range."""
    min_val, max_val = SPEC_RANGES.get(spec, (0, 100))
    
    if value <= min_val:
        return 0.0
    if value >= max_val:
        return 1.0
    return (value - min_val) / (max_val - min_val)

def check_constraint_satisfaction(product, constraint):
    """Check if a product satisfies a single constraint."""
    spec = constraint["spec"]
    operator = constraint["operator"]
    constraint_value = constraint["value"]
    product_value = product["specs"].get(spec)
    
    # Skip if product doesn't have this spec
    if product_value is None:
        return False
    
    # Check constraint based on operator
    if operator == "min":
        return product_value >= constraint_value
    elif operator == "max":
        return product_value <= constraint_value
    elif operator == "is":
        return product_value == constraint_value
    
    # For preference operators, return True (they don't disqualify products)
    return True

def calculate_soft_constraint_score(product, constraint):
    """Calculate score for a single soft constraint."""
    spec = constraint["spec"]
    operator = constraint["operator"]
    product_value = product["specs"].get(spec)
    
    # Skip if product doesn't have this spec
    if product_value is None:
        return 0.0
    
    # Calculate score based on operator
    if operator == "prefer_higher":
        if spec in SPEC_RANGES:
            return normalize_value(product_value, spec)
        return 0.7 if product_value > AVG_SPEC_VALUES.get(spec, 0) else 0.3
    
    elif operator == "prefer_lower":
        if spec in SPEC_RANGES:
            return 1.0 - normalize_value(product_value, spec)
        return 0.7 if product_value < AVG_SPEC_VALUES.get(spec, 0) else 0.3
    
    elif operator == "prefer":
        return 1.0 if product_value == constraint["value"] else 0.2
    
    return 0.5  # Default middle score

def generate_relevance_scores(products, target_product, hard_constraints, soft_constraints):
    """Generate ground truth relevance scores for all products."""
    relevance_scores = []
    threshold_position = None
    product_type = target_product["product_type"]
    
    # First filter to only include products of the matching type
    matching_type_products = [p for p in products if p["product_type"] == product_type]
    
    for product in matching_type_products:
        # Check each hard constraint
        hard_results = {c["spec"]: check_constraint_satisfaction(product, c) for c in hard_constraints}
        meets_all_hard = all(hard_results.values())
        
        # Calculate soft constraint scores
        soft_results = {}
        weighted_sum = 0
        total_weight = 0
        
        for constraint in soft_constraints:
            spec = constraint["spec"]
            weight = constraint.get("weight", 3)
            score = calculate_soft_constraint_score(product, constraint)
            
            soft_results[spec] = score
            weighted_sum += score * weight
            total_weight += weight
        
        # Overall soft score (weighted average)
        soft_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Final score - 0 if hard constraints not met
        overall_score = soft_score if meets_all_hard else 0.0
        
        # Boost score for target product
        if product["id"] == target_product["id"]:
            overall_score = min(1.0, overall_score * 1.1)
        
        relevance_scores.append({
            "product_id": product["id"],
            "product_type": product["product_type"],
            "tier": product["tier"],
            "meets_hard_constraints": meets_all_hard,
            "hard_constraint_results": hard_results,
            "soft_constraint_score": soft_score,
            "soft_constraint_results": soft_results,
            "overall_score": overall_score
        })
    
    # Sort by overall score (descending)
    relevance_scores.sort(key=lambda x: x["overall_score"], reverse=True)
    
    # Find threshold position where products stop meeting hard constraints
    for i, score in enumerate(relevance_scores):
        if not score["meets_hard_constraints"]:
            threshold_position = i
            break
    
    return relevance_scores, threshold_position

def generate_full_ranked_list(products, relevance_scores):
    """Generate full ranked list with threshold indication."""
    # Create a mapping from product_id to relevance score details
    score_map = {score["product_id"]: score for score in relevance_scores}
    
    # Generate full ranked list with all products
    full_ranked_list = []
    for i, product_id in enumerate([score["product_id"] for score in relevance_scores]):
        full_ranked_list.append({
            "rank": i + 1,
            "product_id": product_id,
            "score": score_map[product_id]["overall_score"],
            "meets_constraints": score_map[product_id]["meets_hard_constraints"]
        })
    
    return full_ranked_list

def generate_requests(llm_client, products, num_requests=CONFIG["num_requests"]):
    """Generate search queries with constraints and filtered rankings."""
    generated_requests = []
    
    # Select target products randomly
    target_indices = random.sample(range(len(products)), min(num_requests, len(products)))
    
    print(f"\nGenerating {len(target_indices)} search queries...")
    
    for i, idx in enumerate(tqdm(target_indices, desc="Generating queries")):
        target_product = products[idx]
        query_type = random.choice(QUERY_TYPES)
        
        print(f"\n[{i+1}/{len(target_indices)}] Generating {query_type['name']} query for {target_product['product_type']} (Tier: {target_product['tier']})")
        
        # Generate constraints
        hard_constraints, soft_constraints = select_constraints(
            target_product,
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
        
        # Generate search query
        query_text = generate_query(
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
        
        # Generate relevance scores and filtering
        relevance_scores, threshold_position = generate_relevance_scores(
            products, 
            target_product, 
            hard_constraints, 
            soft_constraints
        )
        
        # Filter to only products meeting hard constraints and get their IDs
        valid_items = [score for score in relevance_scores if score["meets_hard_constraints"]]
        ranked_list = [item["product_id"] for item in valid_items]
        
        # Create request
        request = {
            "id": len(generated_requests),
            "query": query_text,
            "target_product_id": target_product["id"],
            "product_type": target_product["product_type"],
            "tier": target_product["tier"],
            "query_type": query_type["name"],
            "hard_constraints": hard_constraints,
            "soft_constraints": soft_constraints,
            "ranked_list": ranked_list,  # Only include valid items
            "threshold_position": threshold_position
        }
        
        generated_requests.append(request)
        
        # Delay between requests
        if i < len(target_indices) - 1 and CONFIG["llm_delay"] > 0:
            time.sleep(CONFIG["llm_delay"])
    
    return generated_requests

def main():
    """Main execution function."""
    # Create output directory
    ensure_dir(CONFIG["output_dir"])
    
    # Initialize LLM client
    llm_client = create_llm_client()
    print("LLM Client initialized successfully.")
    
    product_path = Path(CONFIG["output_dir"]) / CONFIG["filenames"]["products"]
    
    # Check if products already exist
    if product_path.exists():
        print(f"Loading existing products from {product_path}")
        with open(product_path, 'r') as f:
            import json
            products = json.load(f)
    else:
        # Generate products
        products = generate_products(llm_client, CONFIG["num_products"])
        
        # Save products
        dumpj(products, product_path)
        print(f"Generated and saved {len(products)} products to {product_path}")
    
    # Generate requests with filtered rankings
    requests = generate_requests(llm_client, products, CONFIG["num_requests"])

    # Save requests
    requests_path = Path(CONFIG["output_dir"]) / CONFIG["filenames"]["requests"]
    dumpj(requests, requests_path)
    print(f"Generated and saved {len(requests)} requests to {requests_path}")

    # Print summary
    print("\nDataset Generation Complete")
    print(f"Products: {len(products)}")
    print(f"Requests: {len(requests)}")
    print(f"All files saved to directory: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()