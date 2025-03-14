import logging
import re
import json
from utils import user_struct, system_struct, assistant_struct

from prompts import (
    SYSTEM_PROMPT, 
    QUERY_PROMPT, 
    AVAILABLE_ACTIONS, 
    get_state_info, 
    REASONING_PROMPT,
    ACTION_SELECTION_PROMPT
)

class ReactAgent:
    """
    A ReAct agent that can search for items in a file system using LLM guidance.
    """
    
    def __init__(self, file_system, llm_client):
        """Initialize the agent with a file system and LLM client."""
        self.fs = file_system
        self.llm_client = llm_client
        
    def search(self, query, max_steps=5):
        """
        Search for items matching the query using a multi-step reasoning process.
        Returns a dict with results and search information.
        """
        # Initialize search state
        state = {
            "query": query,
            "current_path": "/",
            "steps": [],
            "found_items": set(),
            "step_count": 0,
            "complete": False
        }
        
        # Start conversation with system prompt and query
        base_messages = [
            system_struct(SYSTEM_PROMPT),
            user_struct(QUERY_PROMPT.format(query=query))
        ]
        
        # Main ReAct loop
        while not state["complete"] and state["step_count"] < max_steps:
            # Get current file system information
            fs_info = self.fs.navigate_to(state["current_path"])
            
            # FIRST LLM CALL: Get reasoning
            # Add information about current state to messages
            state_info = get_state_info(state, fs_info, max_steps)
            reasoning_messages = base_messages.copy()
            reasoning_messages.append(system_struct(state_info))
            reasoning_messages.append(user_struct(REASONING_PROMPT))
            
            print("\n" + "="*80)
            print(f"STEP {state['step_count'] + 1}: REASONING INPUT")
            print("="*80)
            for msg in reasoning_messages:  # Print the last two messages
                print(f"[{msg['role'].upper()}]")
                print(msg['content'])
                print("-"*40)
            
            reasoning = self.llm_client(reasoning_messages)
            
            print("\n" + "="*80)
            print(f"STEP {state['step_count'] + 1}: REASONING OUTPUT")
            print("="*80)
            print(reasoning)
            
            # SECOND LLM CALL: Get action
            action_messages = reasoning_messages.copy()
            action_messages.append(assistant_struct(reasoning))
            action_messages.append(user_struct(ACTION_SELECTION_PROMPT))
            
            print("\n" + "="*80)
            print(f"STEP {state['step_count'] + 1}: ACTION INPUT")
            print("="*80)
            print(f"[USER]\n{ACTION_SELECTION_PROMPT}")
            
            action = self.llm_client(action_messages)
            
            print("\n" + "="*80)
            print(f"STEP {state['step_count'] + 1}: ACTION OUTPUT")
            print("="*80)
            print(action)
            
            # Update base_messages with reasoning and action for context in next iterations
            base_messages.append(system_struct(state_info))
            base_messages.append(assistant_struct(f"Reasoning: {reasoning}\n\nAction: {action}"))
            
            # Parse and execute the action
            action_result = self._execute_action(action, state, reasoning)
            base_messages.append(system_struct(action_result))
            
            print("\n" + "="*80)
            print(f"STEP {state['step_count'] + 1}: ACTION RESULT")
            print("="*80)
            print(action_result)
            
            # Add breakpoint
            input(f"\nPress Enter to continue to step {state['step_count'] + 2}...")
            
            # Increment step count
            state["step_count"] += 1
        
        # Prepare final results
        if state["complete"]:
            results = {
                "success": True,
                "items": [self.fs.get_item_by_id(item_id) for item_id in state["found_items"]],
                "steps": state["steps"],
                "summary": f"Found {len(state['found_items'])} items in {state['step_count']} steps."
            }
        else:
            results = {
                "success": False,
                "items": [],
                "steps": state["steps"],
                "message": f"Search incomplete after {max_steps} steps."
            }
        
        return results
    
    def _execute_action(self, action, state, reasoning=None):
        """Parse and execute the action from the LLM response."""
        # Record the reasoning
        state["steps"].append({"reasoning": reasoning or action, "action": action})
        
        # Parse action using regex
        navigate_match = re.search(r'Navigate\s*\(\s*["\']?([^)"\']+)["\']?\s*\)', action)
        search_match = re.search(r'Search\s*\(\s*["\']?([^)"\']+)["\']?\s*\)', action)
        subcategory_match = re.search(r'CreateSubcategory\s*\(\s*["\']?([^,"\']+)["\']?\s*,\s*\[([^\]]+)\]\s*\)', action)
        tag_match = re.search(r'AddTag\s*\(\s*\[([^\]]+)\]\s*,\s*["\']?([^)"\']+)["\']?\s*\)', action)
        remove_tag_match = re.search(r'RemoveTag\s*\(\s*\[([^\]]+)\]\s*,\s*["\']?([^)"\']+)["\']?\s*\)', action)
        get_tag_match = re.search(r'GetByTag\s*\(\s*["\']?([^)"\']+)["\']?\s*\)', action)
        complete_match = re.search(r'Complete\s*\(\s*\[([^\]]+)\]\s*\)', action)
        
        # Execute the action
        if navigate_match:
            path = navigate_match.group(1)
            return self._navigate(path, state)
        elif search_match:
            keywords = search_match.group(1)
            return self._search(keywords, state)
        elif subcategory_match:
            name = subcategory_match.group(1)
            item_ids_str = subcategory_match.group(2)
            return self._create_subcategory(name, item_ids_str, state)
        elif tag_match:
            item_ids_str = tag_match.group(1)
            tag = tag_match.group(2)
            return self._add_tag(item_ids_str, tag, state)
        elif remove_tag_match:
            item_ids_str = remove_tag_match.group(1)
            tag = remove_tag_match.group(2)
            return self._remove_tag(item_ids_str, tag, state)
        elif get_tag_match:
            tag = get_tag_match.group(1)
            return self._get_by_tag(tag, state)
        elif complete_match:
            item_ids_str = complete_match.group(1)
            return self._complete(item_ids_str, state)
        else:
            return "Could not parse action. Please use one of the available actions with the correct syntax."
    
    def _navigate(self, path, state):
        """Navigate to a specific category path."""
        if not path.startswith('/'):
            path = f"/{path}"
        
        fs_info = self.fs.navigate_to(path)
        if fs_info['total_items'] > 0:
            state["current_path"] = path
            result = f"Successfully navigated to {path}."
        else:
            result = f"Path {path} not found or contains no items."
        
        state["steps"].append({"action": "Navigate", "path": path, "result": result})
        return result
    
    def _search(self, keywords_str, state):
        """Search for items containing any of the keywords."""
        keywords = [kw.strip() for kw in keywords_str.split(',')]
        items = self.fs.keyword_search(keywords, state["current_path"])
        
        result = f"Found {len(items)} items matching keywords: {keywords_str}"
        if items:
            # Add sample items
            sample_size = min(3, len(items))
            samples = []
            for item_id in items[:sample_size]:
                item = self.fs.get_item_by_id(item_id)
                if item:
                    samples.append({
                        "item_id": item_id,
                        "metadata": item.get('metadata', '')[:100] + "..." if len(item.get('metadata', '')) > 100 else item.get('metadata', '')
                    })
            
            result += f"\nSample items:\n"
            for sample in samples:
                result += f"- {sample['item_id']}: {sample['metadata']}\n"
            
            # Add all item_ids
            result += f"\nAll item_ids: {items[:10]}"
            if len(items) > 10:
                result += f" and {len(items) - 10} more."
        
        state["steps"].append({"action": "Search", "keywords": keywords, "result_count": len(items), "items": items})
        return result
    
    def _create_subcategory(self, name, item_ids_str, state):
        """Create a new subcategory with the specified items."""
        item_ids = [id.strip().strip('"\'') for id in item_ids_str.split(',')]
        moved_count = self.fs.create_subcategory(state["current_path"], name, item_ids)
        
        result = f"Created subcategory '{name}' with {moved_count} items."
        state["steps"].append({"action": "CreateSubcategory", "name": name, "item_count": moved_count})
        return result
    
    def _add_tag(self, item_ids_str, tag, state):
        """Add a tag to the specified items."""
        item_ids = [id.strip().strip('"\'') for id in item_ids_str.split(',')]
        tagged_count = self.fs.add_tag(item_ids, tag)
        
        result = f"Added tag '{tag}' to {tagged_count} items."
        state["steps"].append({"action": "AddTag", "tag": tag, "item_count": tagged_count})
        return result
    
    def _remove_tag(self, item_ids_str, tag, state):
        """Remove a tag from the specified items."""
        item_ids = [id.strip().strip('"\'') for id in item_ids_str.split(',')]
        removed_count = self.fs.remove_tag(item_ids, tag)
        
        result = f"Removed tag '{tag}' from {removed_count} items."
        state["steps"].append({"action": "RemoveTag", "tag": tag, "item_count": removed_count})
        return result
    
    def _get_by_tag(self, tag, state):
        """Get items with the specified tag."""
        items = self.fs.get_items_by_tag(tag, state["current_path"])
        
        result = f"Found {len(items)} items with tag '{tag}'."
        if items:
            # Add sample items
            sample_size = min(3, len(items))
            samples = []
            for item_id in items[:sample_size]:
                item = self.fs.get_item_by_id(item_id)
                if item:
                    samples.append({
                        "item_id": item_id,
                        "metadata": item.get('metadata', '')[:100] + "..." if len(item.get('metadata', '')) > 100 else item.get('metadata', '')
                    })
            
            result += f"\nSample items:\n"
            for sample in samples:
                result += f"- {sample['item_id']}: {sample['metadata']}\n"
            
            # Add all item_ids
            result += f"\nAll item_ids: {items[:10]}"
            if len(items) > 10:
                result += f" and {len(items) - 10} more."
        
        state["steps"].append({"action": "GetByTag", "tag": tag, "result_count": len(items), "items": items})
        return result
    
    def _complete(self, item_ids_str, state):
        """Finish the search and return the found items."""
        item_ids = [id.strip().strip('"\'') for id in item_ids_str.split(',')]
        state["found_items"] = set(item_ids)
        state["complete"] = True
        
        result = f"Search completed with {len(state['found_items'])} items."
        state["steps"].append({"action": "Complete", "item_count": len(state["found_items"])})
        return result
    
    def save_file_system(self, path):
        """Save the current state of the file system."""
        self.fs.save(path)