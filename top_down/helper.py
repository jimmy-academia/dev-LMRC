

def create_output_report(results, output_path):
    """Create a detailed HTML report of search results."""
    if not results["success"]:
        return
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results - {results["summary"][:50]}...</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .item {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .item:hover {{ background-color: #f9f9f9; }}
            .tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0; }}
            .tag {{ background-color: #eee; padding: 3px 8px; border-radius: 10px; font-size: 12px; }}
            .path {{ color: #666; font-style: italic; }}
            .item-id {{ color: #999; font-size: 12px; }}
            h1 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .requirements {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>{results["summary"]}</p>
        </div>
        
        <div class="requirements">
            <h2>Search Requirements</h2>
            <ul>
    """
    
    for req in results["requirements"]:
        html_content += f"<li>{req}</li>\n"
    
    html_content += """
            </ul>
        </div>
        
        <h2>Top Results</h2>
    """
    
    for item in results["items"]:
        # Format tags
        tags_html = ""
        if "tags" in item:
            for tag in item["tags"]:
                tags_html += f'<span class="tag">{tag}</span>'
        
        html_content += f"""
        <div class="item">
            <h3>{item["metadata"][:100]}...</h3>
            <div class="item-id">ID: {item["item_id"]}</div>
            <div class="path">Path: {item.get("path", item["category"])}</div>
            <div class="tags">{tags_html}</div>
            <p>{item["metadata"]}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Output report saved to {output_path}")

