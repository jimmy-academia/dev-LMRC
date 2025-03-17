import logging
import json
import re

from utils import user_struct, system_struct, assistant_struct
from prompts import (
    SYSTEM_PROMPT,
    QUERY_PROMPT,
    SINGLE_STEP_PROMPT,
    get_state_info,
)

class ReactAgent:
    """
    A ReAct agent that can search for items in a file system using LLM guidance,
    but refactored to use one single LLM call (Reasoning + Action) per step
    and parse it as JSON.
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

        # Prepare the initial conversation
        base_messages = [
            system_struct(SYSTEM_PROMPT),
            user_struct(QUERY_PROMPT.format(query=query))
        ]

        # Main ReAct loop
        while not state["complete"] and state["step_count"] < max_steps:
            # 1) Get current file system information
            fs_info = self.fs.navigate_to(state["current_path"])
            state_info = get_state_info(state, fs_info, max_steps)

            # 2) Build single-step prompt that requests both reasoning + action
            step_prompt = SINGLE_STEP_PROMPT.format(query=query, state_info=state_info)

            # 3) Prepare messages
            step_messages = base_messages.copy()
            step_messages.append(user_struct(step_prompt))

            # 4) Single LLM call for reasoning + action
            llm_output = self.llm_client(step_messages)

            # For debugging/logging, print out the raw LLM response:
            print(f"\n--- STEP {state['step_count']+1} RAW OUTPUT ---\n{llm_output}\n")

            # 5) Parse the LLM's JSON output
            parsed_output = self._parse_llm_json_output(llm_output)
            if not parsed_output:
                # If parsing failed, log error & break (or you could retry)
                error_message = "ERROR: Could not parse LLM JSON output."
                logging.error(error_message)
                state["steps"].append({"error": error_message, "raw_output": llm_output})
                break

            # Extract reasoning and action
            reasoning = parsed_output.get("reasoning", "")
            action_dict = parsed_output.get("action", {})
            action_name = action_dict.get("name", "")
            action_params = action_dict.get("params", {})

            # Store the step for debugging or review later
            step_record = {
                "reasoning": reasoning,
                "action": action_dict,
            }
            state["steps"].append(step_record)

            # 6) Execute the parsed action
            action_result = self._execute_action(action_name, action_params, state)
            step_record["result"] = action_result

            # 7) Add the LLM’s output & result to the conversation context
            base_messages.append(assistant_struct(json.dumps(parsed_output)))
            base_messages.append(system_struct(action_result))

            # 8) Increment step
            state["step_count"] += 1

            input(f'pause before step {state["step_count"]}')

        # 9) Prepare final results
        return self._prepare_final_results(state)

    def _parse_llm_json_output(self, llm_text):
        """
        Attempts to parse the LLM output as JSON with a structure:
        {
          "reasoning": "...",
          "action": {
            "name": "<action_name>",
            "params": { ... }
          }
        }
        Returns a dict or None if parsing fails.
        """
        try:
            return json.loads(llm_text.strip())
        except json.JSONDecodeError:
            return None

    def _execute_action(self, action_name, params, state):
        """
        Dispatch to the appropriate method based on 'action_name'.
        """
        if action_name == "Navigate":
            path = params.get("path", "/")
            return self._navigate(path, state)

        elif action_name == "Search":
            keywords = params.get("keywords", [])
            return self._search(keywords, state)

        elif action_name == "CreateSubcategory":
            name = params.get("name", "untitled")
            item_ids = params.get("item_ids", [])
            return self._create_subcategory(name, item_ids, state)

        elif action_name == "AddTag":
            tag = params.get("tag", "")
            item_ids = params.get("item_ids", [])
            return self._add_tag(item_ids, tag, state)

        elif action_name == "RemoveTag":
            tag = params.get("tag", "")
            item_ids = params.get("item_ids", [])
            return self._remove_tag(item_ids, tag, state)

        elif action_name == "GetByTag":
            tag = params.get("tag", "")
            return self._get_by_tag(tag, state)

        elif action_name == "Complete":
            item_ids = params.get("item_ids", [])
            return self._complete(item_ids, state)

        else:
            return f"Unrecognized action '{action_name}'. Must be one of the available actions."

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

        return result

    def _search(self, keywords, state):
        """Search for items containing any of the keywords."""
        items = self.fs.keyword_search(keywords, state["current_path"])

        result = f"Found {len(items)} items matching keywords: {keywords}."
        if items:
            sample_items = items[:3]
            result += f" Example item IDs: {sample_items}"

        return result

    def _create_subcategory(self, name, item_ids, state):
        """Create a new subcategory with the specified items."""
        moved_count = self.fs.create_subcategory(state["current_path"], name, item_ids)
        return f"Created subcategory '{name}' with {moved_count} items."

    def _add_tag(self, item_ids, tag, state):
        """Add a tag to the specified items."""
        tagged_count = self.fs.add_tag(item_ids, tag)
        return f"Added tag '{tag}' to {tagged_count} items."

    def _remove_tag(self, item_ids, tag, state):
        """Remove a tag from the specified items."""
        removed_count = self.fs.remove_tag(item_ids, tag)
        return f"Removed tag '{tag}' from {removed_count} items."

    def _get_by_tag(self, tag, state):
        """Get items with the specified tag."""
        items = self.fs.get_items_by_tag(tag, state["current_path"])

        result = f"Found {len(items)} items with tag '{tag}'."
        if items:
            sample_items = items[:3]
            result += f" Example item IDs: {sample_items}"

        return result

    def _complete(self, item_ids, state):
        """Finish the search and record the found items."""
        state["found_items"] = set(item_ids)
        state["complete"] = True
        return f"Search completed with {len(state['found_items'])} items."

    def _prepare_final_results(self, state):
        """Build the final result object after the search loop finishes."""
        if state["complete"]:
            items = [self.fs.get_item_by_id(item_id) for item_id in state["found_items"]]
            return {
                "success": True,
                "items": items,
                "steps": state["steps"],
                "summary": f"Found {len(state['found_items'])} items in {state['step_count']} steps."
            }
        else:
            return {
                "success": False,
                "items": [],
                "steps": state["steps"],
                "message": f"Search incomplete after {state['step_count']} steps."
            }

    def save_file_system(self, path):
        """Save the current state of the file system."""
        self.fs.save(path)
