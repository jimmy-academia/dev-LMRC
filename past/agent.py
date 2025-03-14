from prompts import LLMPrompts

class ReactAgent:
    """Agent that uses Reasoning, Action, Observation loop for search."""
    def __init__(self, fs, llm_client):
        # Data sources
        self.fs = fs
        self.llm_client = llm_client
        self.prompts = LLMPrompts()
        
        # Search state tracking
        self.enhanced_index = None  # Will be set later with cached version
        self.search_results = []
        self.current_query = ""
        self.current_reasoning = ""
        
        # Search context state
        self.context = {
            "requirements": [],
            "current_path": "/",
            "explored_paths": set(),
            "relevant_tags": set(),
            "previously_viewed_items": set(),
            "category_insights": {}  # Maps category -> observations
        }
    
    def search(self, query, max_steps=10):
        """Run the REACT loop for a search query."""
        self.current_query = query
        
        # Reset search state
        self.search_results = []
        self.context = {
            "requirements": [],
            "current_path": "/",
            "explored_paths": set(),
            "relevant_tags": set(),
            "previously_viewed_items": set(),
            "category_insights": {},  # Maps category -> observations
            "previous_actions": []    # Track previous actions with results
        }
        
        # Extract search requirements using LLM
        self._analyze_requirements(query)
        
        # Begin REACT loop
        for step in range(max_steps):
            print(f"\n=== Step {step+1}/{max_steps} ===")
            
            # 1. Reasoning - Decide what to do next
            reasoning = self._reasoning_step()
            self.current_reasoning = reasoning
            print(f"Reasoning:\n{reasoning}")  # Print full reasoning
            
            # 2. Action - Execute the chosen action
            action = self._determine_next_action(reasoning)
            print(f"Action: {action['type']}")
            print(f"Params: {action['params']}")
            
            # 3. Observation - Record the results
            observation = self._execute_action(action)
            print(f"Observation:\n{observation}")  # Print full observation
            
            # Check if search is complete
            if "SEARCH_COMPLETE" in action['type']:
                break
        
        # Return final, organized results
        return self._finalize_results()
    
    def _analyze_requirements(self, query):
        """Extract key requirements from the search query using LLM."""
        # Get requirements prompt
        prompt = self.prompts.get_requirements_prompt(query)
        
        response = self.llm_client(prompt)
        
        # Extract requirements from bullet points
        requirements = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                requirements.append(line[2:])
            elif line.startswith('•'):
                requirements.append(line[1:].strip())
        
        self.context["requirements"] = requirements
        
        # Also determine likely categories
        category_prompt = self.prompts.get_category_prompt(query, self.enhanced_index.get_all_categories())
        
        category_response = self.llm_client(category_prompt)
        suggested_categories = [c.strip() for c in category_response.split(',')]
        
        # Store the top categories for exploration
        self.context["top_categories"] = suggested_categories[:3]
    
    def _reasoning_step(self):
        """Reason about the next best action based on current state."""
        # Format the current state
        current_path = self.context["current_path"]
        
        # Get items at current path
        items_in_path = self.fs.get_items_by_path(current_path)
        
        # Get a sample of items if we have some
        sample_items_text = ""
        if items_in_path:
            sample_size = min(3, len(items_in_path))
            sample_ids = random.sample(items_in_path, sample_size)
            
            samples = []
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                tags = ", ".join(item.tags) if item.tags else "No tags"
                samples.append(f"Sample {i+1}:\nPath: {item.path}\nTags: {tags}\nDesc: {item.raw_metadata[:150]}...")
            
            sample_items_text = "\n\n".join(samples)
        
        # Format requirements
        requirements_text = "\n".join([f"- {req}" for req in self.context["requirements"]])
        
        # Format recent search results if any
        recent_results = ""
        if self.search_results:
            recent_size = min(3, len(self.search_results))
            recent_samples = []
            for i, idx in enumerate(self.search_results[:recent_size]):
                item_id = self.item_pool[idx].get('item_id')
                if item_id in self.fs.items:
                    item = self.fs.items[item_id]
                    tags = ", ".join(item.tags) if item.tags else "No tags"
                    recent_samples.append(f"Result {i+1}:\nPath: {item.path}\nTags: {tags}\nDesc: {item.raw_metadata[:150]}...")
            
            recent_results = "\n\n".join(recent_samples)
        
        # Get subdirectories
        subdirs = self.fs.get_subdirectories(current_path)
        
        # Get the reasoning prompt
        prompt = self.prompts.get_reasoning_prompt(
            self.current_query,
            requirements_text,
            current_path,
            len(items_in_path),
            subdirs,
            self.context['relevant_tags'],
            recent_results,
            sample_items_text
        )
        
        return self.llm_client(prompt, max_tokens=800)
    
    def _determine_next_action(self, reasoning):
        """Determine the next action based on the reasoning."""
        # Potential actions the agent can take
        actions = [
            "NAVIGATE_TO_CATEGORY",
            "NAVIGATE_TO_PATH",
            "TEXT_SEARCH",
            "CREATE_SUBCATEGORIES",
            "ASSIGN_TAGS",
            "FILTER_BY_TAGS",
            "EVALUATE_ITEMS",
            "SEARCH_COMPLETE"
        ]
        
        # Get action prompt
        action_prompt = self.prompts.get_action_prompt(reasoning, actions)
        
        action_response = self.llm_client(action_prompt, max_tokens=200)
        
        # Parse the action response
        action_type = None
        action_params = None
        
        for line in action_response.strip().split('\n'):
            line = line.strip()
            if line.startswith('ACTION:'):
                action_type = line[7:].strip()
            elif line.startswith('PARAMS:'):
                action_params = line[7:].strip()
        
        # Default to EVALUATE_ITEMS if parsing failed
        if not action_type:
            for action in actions:
                if action.lower() in reasoning.lower():
                    action_type = action
                    break
            
            if not action_type:
                action_type = "EVALUATE_ITEMS"
        
        return {
            "type": action_type,
            "params": action_params
        }
    
    def _execute_action(self, action):
        """Execute the determined action and return observations."""
        action_type = action["type"]
        params = action["params"]
        
        # Record this action
        action_record = {
            "type": action_type,
            "params": params
        }
        
        # Add to previous actions
        if "previous_actions" not in self.context:
            self.context["previous_actions"] = []
        
        result = None
        if action_type == "NAVIGATE_TO_CATEGORY":
            result = self._action_navigate_to_category(params)
        elif action_type == "NAVIGATE_TO_PATH":
            result = self._action_navigate_to_path(params)
        elif action_type == "TEXT_SEARCH":
            result = self._action_text_search(params)
            # Check if we got zero results
            if "found 0 results" in result:
                action_record["result_count"] = 0
                
                # Include search strategy advice if needed
                if params and ',' in params:
                    keywords = [k.strip() for k in params.split(',')]
                    if len(keywords) > 2:
                        # Add suggestion for narrower search
                        result += "\n\nSEARCH STRATEGY ADVICE: Your search used multiple keywords: " + params
                        result += "\nThis search method requires documents to contain ALL keywords."
                        result += "\nTry searching with fewer keywords, such as:"
                        for i in range(min(3, len(keywords))):
                            result += f"\n- '{keywords[i]}'"
            else:
                # Extract result count from the text
                match = re.search(r'found (\d+) results', result)
                if match:
                    action_record["result_count"] = int(match.group(1))
        elif action_type == "CREATE_SUBCATEGORIES":
            result = self._action_create_subcategories()
        elif action_type == "ORGANIZE_SEARCH_RESULTS":
            result = self._action_organize_search_results(params)
        elif action_type == "ASSIGN_TAGS":
            result = self._action_assign_tags()
        elif action_type == "FILTER_BY_TAGS":
            result = self._action_filter_by_tags(params)
        elif action_type == "EVALUATE_ITEMS":
            result = self._action_evaluate_items()
        elif action_type == "SEARCH_COMPLETE":
            result = "Search complete. Finalizing results."
        else:
            result = f"Unknown action type: {action_type}"
        
        # Save the result
        action_record["result"] = result
        self.context["previous_actions"].append(action_record)
            
        return result
            
    def _action_organize_search_results(self, subcategory_path):
        """Organize current search results into specified subcategory path."""
        if not self.search_results:
            return "No search results to organize."
            
        if not subcategory_path:
            # Ask LLM to suggest a path
            prompt = self.prompts.get_organize_results_prompt()
            subcategory_path = self.llm_client(prompt).strip()
            
        # Normalize path (remove leading slash if present)
        if subcategory_path.startswith('/'):
            subcategory_path = subcategory_path[1:]
            
        # Extract the base category from the path
        path_parts = subcategory_path.split('/')
        base_category = path_parts[0]
        
        # Check if base category exists
        categories = self.enhanced_index.get_all_categories()
        if base_category not in categories:
            return f"Base category '{base_category}' not found. Available categories: {', '.join(categories[:5])}..."
            
        # Get item IDs from search results
        organized_count = 0
        for idx in self.search_results:
            item = self.item_pool[idx]
            item_id = item.get('item_id')
            
            if item_id in self.fs.items:
                # Update the path
                full_path = '/' + subcategory_path
                self.fs.organize_item(item_id, full_path)
                organized_count += 1
                
        # Sample some organized items to show
        items_in_path = self.fs.get_items_by_path('/' + subcategory_path)
        sample_size = min(3, len(items_in_path))
        
        if sample_size > 0:
            sample_ids = random.sample(items_in_path, sample_size)
            samples = []
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                samples.append(f"Sample {i+1}: {item.raw_metadata[:150]}...")
            samples_text = "\n".join(samples)
        else:
            samples_text = "No items in this path."
            
        return f"Organized {organized_count} search results into path '/{subcategory_path}'.\n\n{samples_text}"
    
    def _action_navigate_to_category(self, category_name):
        """Navigate to a root category."""
        if not category_name:
            # Try to pick from top categories if available
            if "top_categories" in self.context and self.context["top_categories"]:
                category_name = self.context["top_categories"][0]
            else:
                # Default to first category in the index
                categories = self.enhanced_index.get_all_categories()
                if categories:
                    category_name = categories[0]
                else:
                    return "No category specified and no categories available."
        
        # Handle case where LLM suggests multiple categories (comma-separated or "and" separated)
        if ',' in category_name or ' and ' in category_name:
            # Split into multiple categories
            multi_categories = []
            for separator in [',', ' and ']:
                if separator in category_name:
                    multi_categories.extend([c.strip() for c in category_name.split(separator)])
            
            if multi_categories:
                category_name = multi_categories[0]  # Use the first one for navigation
                print(f"Multiple categories detected: {multi_categories}")
                print(f"Navigating to first category: {category_name}")
                # Store others for potential exploration
                self.context["other_categories"] = multi_categories[1:]
        
        # Normalize category name (match case with available categories)
        available_categories = self.enhanced_index.get_all_categories()
        matched_category = None
        for avail_cat in available_categories:
            if avail_cat.lower() == category_name.lower():
                matched_category = avail_cat
                break
        
        if matched_category:
            category_name = matched_category
        else:
            # Try to find the best matching category
            print(f"Couldn't find exact match for category '{category_name}'")
            best_match = None
            best_score = 0
            
            for avail_cat in available_categories:
                # Simple string matching
                if category_name.lower() in avail_cat.lower() or avail_cat.lower() in category_name.lower():
                    score = len(set(category_name.lower()) & set(avail_cat.lower()))
                    if score > best_score:
                        best_score = score
                        best_match = avail_cat
            
            if best_match:
                print(f"Using closest matching category: '{best_match}'")
                category_name = best_match
            else:
                return f"Could not find category '{category_name}' or any close match."
        
        # Update current path
        self.context["current_path"] = f"/{category_name}"
        self.context["explored_paths"].add(self.context["current_path"])
        
        # Get items in this category
        items = self.fs.get_items_by_path(self.context["current_path"])
        
        # Get and analyze subcategories
        subcategory_info = self._analyze_subdirectories(self.context["current_path"])
        
        # Debug info
        print(f"Navigating to /{category_name}")
        print(f"Found {len(items)} items and {len(subcategory_info)} subcategories")
        print(f"Directory structure: {len(self.fs.directories)} paths and {len(self.fs.items)} total items")
        print(f"Category paths: {[path for path in self.fs.directories.keys() if category_name in path][:5]}")
        
        # Sample a few items to show
        sample_text = ""
        if items:
            sample_size = min(5, len(items))
            sample_ids = random.sample(items, sample_size)
            samples = []
            for i, item_id in enumerate(sample_ids):
                if item_id in self.fs.items:
                    item = self.fs.items[item_id]
                    samples.append(f"Sample {i+1}: {item.raw_metadata}")
            sample_text = "\n".join(samples)
        else:
            # Check if category has any items
            category_items = [item_id for item_id, item in self.fs.items.items() if item.category == category_name]
            if category_items:
                sample_text = f"Category has {len(category_items)} items, but none found at path /{category_name}. This may be a file system issue."
            else:
                sample_text = "No items in this category."
        
        # Create response with subcategory information
        subcategory_text = ""
        if subcategory_info:
            subcategory_text = "\nSubcategories available:\n"
            for subcat, count in subcategory_info:
                subcategory_text += f"- {subcat} ({count} items)\n"
        else:
            subcategory_text = "\nNo subcategories available at this level."
        
        return f"Navigated to category /{category_name}\nItems: {len(items)}\n{subcategory_text}\n\n{sample_text}"
        
    def _analyze_subdirectories(self, path):
        """Analyze what subdirectories exist and how many items they contain."""
        subdirs = self.fs.get_subdirectories(path)
        if not subdirs:
            # Try to find any paths that would be child paths of the current path
            child_paths = []
            for directory in self.fs.directories.keys():
                if directory.startswith(path + '/'):
                    # Extract the next component of the path
                    remaining = directory[len(path) + 1:]
                    if '/' in remaining:
                        next_part = remaining.split('/')[0]
                        child_paths.append(path + '/' + next_part)
                    else:
                        child_paths.append(directory)
            
            # Remove duplicates
            subdirs = list(set(child_paths))
        
        # Get item counts for each subdirectory
        subdir_info = []
        for subdir in subdirs:
            # Extract just the name of the subdirectory
            name = subdir.split('/')[-1] if '/' in subdir else subdir
            items = self.fs.get_items_by_path(subdir)
            subdir_info.append((name, len(items)))
        
        return sorted(subdir_info, key=lambda x: x[1], reverse=True)
    
    def _action_navigate_to_path(self, path):
        """Navigate to a specific path in the file system."""
        # Handle relative paths
        if not path.startswith('/'):
            current = self.context["current_path"]
            if path == "..":
                # Go up one level
                parts = current.strip('/').split('/')
                if len(parts) > 1:
                    path = '/' + '/'.join(parts[:-1])
                else:
                    path = '/'
            else:
                # Append to current path
                path = current.rstrip('/') + '/' + path
        
        # Update current path
        self.context["current_path"] = path
        self.context["explored_paths"].add(path)
        
        # Get items and subdirectories
        items = self.fs.get_items_by_path(path)
        subdirs = self.fs.get_subdirectories(path)
        
        # Sample items
        sample_size = min(3, len(items))
        if sample_size > 0:
            sample_ids = random.sample(items, sample_size)
            samples = []
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                samples.append(f"Sample {i+1}: {item.raw_metadata[:150]}...")
            samples_text = "\n".join(samples)
        else:
            samples_text = "No items in this path."
        
        return f"Navigated to {path}\nItems: {len(items)}\nSubdirectories: {len(subdirs)}\n\n{samples_text}"
    
    def _action_text_search(self, params):
        """Perform a text search, optionally scoped to current path."""
        if not params:
            return "No search query specified."
        
        # Check if this is a union or intersection search
        search_type = "standard"
        if " OR " in params:
            search_type = "union"
            search_parts = [part.strip() for part in params.split(" OR ")]
        elif " AND " in params:
            search_type = "intersection"
            search_parts = [part.strip() for part in params.split(" AND ")]
        else:
            # Standard search
            search_parts = [params]
        
        # Determine if we should scope to current category
        current_path = self.context["current_path"]
        category_filter = None
        if current_path != '/':
            category = current_path.strip('/').split('/')[0]
            category_filter = category
        
        # Keep track of results for each part
        result_count = 0
        search_results = set()
        
        # Perform the search based on search type
        if search_type == "standard":
            search_results = self.enhanced_index.search_by_text(params, category_filter)
            self.search_results = list(search_results)
            result_count = len(search_results)
            
            # If no results, try without category filter
            if not search_results and category_filter:
                search_results = self.enhanced_index.search_by_text(params)
                self.search_results = list(search_results)
                result_count = len(search_results)
        
        elif search_type == "union":
            # Union search - combine results from each term
            print(f"Performing UNION search with terms: {search_parts}")
            for term in search_parts:
                term_results = self.enhanced_index.search_by_text(term, category_filter)
                search_results.update(term_results)
            
            self.search_results = list(search_results)
            result_count = len(search_results)
        
        elif search_type == "intersection":
            # Intersection search - find items that match all terms
            print(f"Performing INTERSECTION search with terms: {search_parts}")
            # Start with the first term's results
            if search_parts:
                search_results = self.enhanced_index.search_by_text(search_parts[0], category_filter)
                # Intersect with each additional term
                for term in search_parts[1:]:
                    term_results = self.enhanced_index.search_by_text(term, category_filter)
                    search_results.intersection_update(term_results)
                
                self.search_results = list(search_results)
                result_count = len(search_results)
        
        # Sample results to show
        sample_size = min(5, result_count)
        samples_text = ""
        
        if sample_size > 0:
            samples = []
            sample_indices = random.sample(self.search_results, sample_size)
            
            for i, idx in enumerate(sample_indices):
                item = self.item_pool[idx]
                item_id = item.get('item_id')
                
                # Add to previously viewed
                self.context["previously_viewed_items"].add(item_id)
                
                # Get path and tags if available
                path = "/"
                tags = []
                if item_id in self.fs.items:
                    fs_item = self.fs.items[item_id]
                    path = fs_item.path
                    tags = list(fs_item.tags)
                
                samples.append(
                    f"Result {i+1}:\nPath: {path}\n" + 
                    (f"Tags: {', '.join(tags)}\n" if tags else "") + 
                    f"Description: {item.get('metadata', '')}"
                )
            
            samples_text = "\n\n".join(samples)
        else:
            samples_text = "No results found."
            
            # Provide search advice based on search type
            if search_type == "standard" and "," in params:
                keywords = [k.strip() for k in params.split(',')]
                if len(keywords) > 1:
                    samples_text += f"\n\nSEARCH STRATEGY ADVICE: Your search used multiple keywords: {params}"
                    samples_text += "\nThis search method requires documents to contain ALL keywords."
                    samples_text += "\nTry searching with fewer keywords, such as:"
                    for i in range(min(3, len(keywords))):
                        samples_text += f"\n- '{keywords[i]}'"
                    samples_text += f"\n\nOr try a UNION search to find documents with ANY of these keywords:"
                    samples_text += f"\n- '{keywords[0]} OR {keywords[1]}'"
            elif search_type == "union" and result_count == 0:
                samples_text += "\n\nEven a union search returned zero results. Try using more generic keywords."
            elif search_type == "intersection" and result_count == 0:
                samples_text += "\n\nThe intersection search is too restrictive. Try using a union search instead."
        
        # Return appropriate search type description
        search_description = params
        if search_type == "union":
            search_description = " OR ".join(search_parts)
        elif search_type == "intersection":
            search_description = " AND ".join(search_parts)
        
        return f"{search_type.capitalize()} text search for '{search_description}' found {result_count} results.\n\n{samples_text}"
    
    def _action_create_subcategories(self):
        """Create subcategories for the current path."""
        current_path = self.context["current_path"]
        
        # Get items in current path
        items = self.fs.get_items_by_path(current_path)
        if not items:
            return "No items in current path to categorize."
        
        # Use LLM to suggest subcategories
        subcategories = self.fs.suggest_subcategories(current_path, self.llm_client)
        
        if not subcategories:
            return "Failed to generate subcategories."
        
        # Sample items to categorize
        sample_size = min(50, len(items))
        sample_ids = random.sample(items, sample_size)
        
        # Process in batches
        batch_size = 5
        batches = [sample_ids[i:i+batch_size] for i in range(0, len(sample_ids), batch_size)]
        
        items_organized = 0
        
        for batch in batches:
            # Format items for the LLM
            batch_text = "\n".join([
                f"Item {i+1}: {self.fs.items[item_id].raw_metadata[:200]}..." 
                for i, item_id in enumerate(batch)
            ])
            
            # Create prompt for categorization
            prompt = self.prompts.get_categorization_prompt(subcategories, batch_text)
            
            response = self.llm_client(prompt)
            
            # Parse and apply classifications
            for line in response.strip().split('\n'):
                if ':' in line and line.strip().upper().startswith('ITEM '):
                    try:
                        item_num = int(line.split(':')[0].strip().upper().replace('ITEM ', '')) - 1
                        if 0 <= item_num < len(batch):
                            subcategory = line.split(':', 1)[1].strip()
                            if subcategory in subcategories:
                                item_id = batch[item_num]
                                new_path = f"{current_path.rstrip('/')}/{subcategory}"
                                self.fs.organize_item(item_id, new_path)
                                items_organized += 1
                    except (ValueError, IndexError):
                        continue
        
        # Organize remaining items by similarity
        if len(items) > sample_size:
            # Collect examples for each subcategory
            examples_by_subcategory = defaultdict(list)
            for item_id in sample_ids:
                item = self.fs.items[item_id]
                parts = item.path.strip('/').split('/')
                if len(parts) > 1:  # Has a subcategory
                    subcategory = parts[-1]
                    examples_by_subcategory[subcategory].append(item)
            
            # Function to find best matching subcategory
            def find_best_subcategory(item):
                item_tokens = set(re.findall(r'\w+', item.raw_metadata.lower()))
                best_score = -1
                best_subcat = None
                
                for subcat, examples in examples_by_subcategory.items():
                    total_score = 0
                    for ex in examples:
                        ex_tokens = set(re.findall(r'\w+', ex.raw_metadata.lower()))
                        score = len(item_tokens.intersection(ex_tokens))
                        total_score += score
                    
                    avg_score = total_score / len(examples) if examples else 0
                    if avg_score > best_score:
                        best_score = avg_score
                        best_subcat = subcat
                
                return best_subcat
            
            # Process remaining items
            remaining_ids = [id for id in items if id not in sample_ids]
            for item_id in remaining_ids:
                item = self.fs.items[item_id]
                best_subcat = find_best_subcategory(item)
                
                if best_subcat:
                    new_path = f"{current_path.rstrip('/')}/{best_subcat}"
                    self.fs.organize_item(item_id, new_path)
                    items_organized += 1
        
        return f"Created {len(subcategories)} subcategories: {', '.join(subcategories)}\nOrganized {items_organized} items into subcategories."
    
    def _action_assign_tags(self):
        """Assign meaningful tags to items in the current view."""
        # Determine which items to tag (search results or current path)
        if self.search_results:
            # Use search results
            items_to_tag = []
            for idx in self.search_results[:50]:  # Limit to first 50 for efficiency
                item = self.item_pool[idx]
                item_id = item.get('item_id')
                if item_id in self.fs.items:
                    items_to_tag.append(item_id)
        else:
            # Use current path
            current_path = self.context["current_path"]
            items_to_tag = self.fs.get_items_by_path(current_path)[:50]
        
        if not items_to_tag:
            return "No items to tag."
        
        # Sample items for tag generation
        sample_size = min(10, len(items_to_tag))
        sample_ids = random.sample(items_to_tag, sample_size)
        
        # Get sample items metadata
        sample_text = ""
        for i, item_id in enumerate(sample_ids):
            item = self.fs.items[item_id]
            sample_text += f"Item {i+1}: {item.raw_metadata[:200]}...\n\n"
        
        # Generate tags using LLM
        tag_prompt = self.prompts.get_tag_generation_prompt(sample_text)
        
        tag_response = self.llm_client(tag_prompt)
        
        # Parse tags
        tags = [tag.strip() for tag in tag_response.split(',')]
        unique_tags = set(tags)
        
        # Now determine which tags apply to which items
        tagged_count = 0
        
        # Process in batches
        batch_size = 5
        batches = [items_to_tag[i:i+batch_size] for i in range(0, len(items_to_tag), batch_size)]
        
        for batch in batches:
            # Format items for the LLM
            batch_text = "\n".join([
                f"Item {i+1}: {self.fs.items[item_id].raw_metadata[:200]}..." 
                for i, item_id in enumerate(batch)
            ])
            
            # Create prompt for tagging
            prompt = self.prompts.get_tag_assignment_prompt(unique_tags, batch_text)
            
            response = self.llm_client(prompt)
            
            # Parse and apply tags
            for line in response.strip().split('\n'):
                if ':' in line and line.strip().upper().startswith('ITEM '):
                    try:
                        item_num = int(line.split(':')[0].strip().upper().replace('ITEM ', '')) - 1
                        if 0 <= item_num < len(batch):
                            item_tags = [t.strip() for t in line.split(':', 1)[1].strip().split(',')]
                            item_id = batch[item_num]
                            
                            # Add valid tags
                            for tag in item_tags:
                                if tag in unique_tags:
                                    self.fs.add_tag(item_id, tag)
                                    # Track relevant tags for search context
                                    self.context["relevant_tags"].add(tag)
                            
                            tagged_count += 1
                    except (ValueError, IndexError):
                        continue
        
        return f"Generated {len(unique_tags)} tags: {', '.join(unique_tags)}\nTagged {tagged_count} items."
    
    def _action_filter_by_tags(self, tags_param):
        """Filter items by tags."""
        if not tags_param:
            # Use relevant tags from context
            tags = list(self.context["relevant_tags"])
            if not tags:
                return "No tags specified for filtering."
        else:
            # Parse tags from parameter
            tags = [t.strip() for t in tags_param.split(',')]
        
        # Get items with these tags
        items = self.fs.get_items_by_tags(tags, require_all=True)
        
        if not items:
            # Try with any match instead of all matches
            items = self.fs.get_items_by_tags(tags, require_all=False)
        
        # Update search results with matching items
        # We need to map item_ids back to indices in the item_pool
        self.search_results = []
        for i, item in enumerate(self.item_pool):
            item_id = item.get('item_id', '')
            if item_id in items:
                self.search_results.append(i)
        
        # Sample results to show
        sample_size = min(5, len(items))
        samples_text = ""
        
        if sample_size > 0:
            sample_ids = random.sample(items, sample_size)
            samples = []
            
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                all_tags = ', '.join(item.tags)
                samples.append(f"Result {i+1}:\nPath: {item.path}\nTags: {all_tags}\nDescription: {item.raw_metadata[:150]}...")
            
            samples_text = "\n\n".join(samples)
        else:
            samples_text = "No matching items found."
        
        return f"Filtered by tags [{', '.join(tags)}]\nFound {len(items)} matching items.\n\n{samples_text}"
    
    def _action_evaluate_items(self):
        """Evaluate current items against search requirements."""
        # Determine which items to evaluate
        items_to_evaluate = []
        
        if self.search_results:
            # Use search results
            for idx in self.search_results[:10]:  # Limit to top 10
                item = self.item_pool[idx]
                item_id = item.get('item_id')
                if item_id in self.fs.items:
                    items_to_evaluate.append((idx, item_id))
        else:
            # Use current path
            current_path = self.context["current_path"]
            path_items = self.fs.get_items_by_path(current_path)
            
            for i, item_id in enumerate(path_items[:10]):  # Limit to top 10
                for idx, item in enumerate(self.item_pool):
                    if item.get('item_id') == item_id:
                        items_to_evaluate.append((idx, item_id))
                        break
        
        if not items_to_evaluate:
            return "No items to evaluate."
        
        # Format requirements
        requirements_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(self.context["requirements"])])
        
        # Evaluate each item
        evaluations = []
        
        for item_index, (idx, item_id) in enumerate(items_to_evaluate):
            item = self.item_pool[idx]
            fs_item = self.fs.items[item_id]
            
            # Format item details
            item_text = f"""
            Item {item_index+1}:
            Category: {fs_item.category}
            Path: {fs_item.path}
            Tags: {', '.join(fs_item.tags)}
            Description: {fs_item.raw_metadata}
            """
            
            # Create evaluation prompt
            prompt = self.prompts.get_evaluation_prompt(requirements_text, item_text)
            
            evaluation = self.llm_client(prompt)
            
            # Extract overall score
            overall_score = 0
            for line in evaluation.strip().split('\n'):
                if line.strip().upper().startswith('OVERALL:'):
                    try:
                        score_text = re.search(r'\((\d+)\)', line)
                        if score_text:
                            overall_score = int(score_text.group(1))
                    except (ValueError, AttributeError):
                        overall_score = 0
            
            # Add to evaluations
            evaluations.append({
                "item_index": idx,
                "item_id": item_id,
                "evaluation": evaluation,
                "overall_score": overall_score
            })
        
        # Sort evaluations by score
        evaluations.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Update search results based on evaluations
        self.search_results = [e["item_index"] for e in evaluations]
        
        # Format evaluation summary
        summary = []
        for i, eval_data in enumerate(evaluations[:5]):  # Show top 5
            item = self.item_pool[eval_data["item_index"]]
            summary.append(f"Item {i+1} (Score: {eval_data['overall_score']}/5):\n" +
                          f"ID: {item.get('item_id')}\n" +
                          f"Description: {item.get('metadata', '')[:100]}...")
        
        return f"Evaluated {len(evaluations)} items against {len(self.context['requirements'])} requirements.\n\n" + "\n\n".join(summary)
    
    def _finalize_results(self):
        """Organize and return final search results."""
        if not self.search_results:
            return {
                "success": False,
                "message": "No matching items found for the query.",
                "items": []
            }
        
        # Limit to top results
        top_results = self.search_results[:20]
        
        # Format results
        results = []
        for idx in top_results:
            item = self.item_pool[idx]
            item_id = item.get('item_id')
            
            result = {
                "item_id": item_id,
                "metadata": item.get('metadata', ''),
                "category": item.get('category', '')
            }
            
            # Add file system organization info if available
            if item_id in self.fs.items:
                fs_item = self.fs.items[item_id]
                result["path"] = fs_item.path
                result["tags"] = list(fs_item.tags)
            
            results.append(result)
        
        # Generate a summary of the search process
        summary_prompt = self.prompts.get_summary_prompt(
            self.current_query,
            self.context["requirements"],
            self.context["explored_paths"],
            self.context["relevant_tags"]
        )
        
        search_summary = self.llm_client(summary_prompt)
        
        return {
            "success": True,
            "message": "Search completed successfully.",
            "summary": search_summary,
            "requirements": self.context["requirements"],
            "relevant_tags": list(self.context["relevant_tags"]),
            "items": results
        }

