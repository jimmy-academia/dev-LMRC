import time
import random
from playwright.sync_api import sync_playwright, TimeoutError

class ChatGPTWebAutomation:
    def __init__(self, headless=True, slow_mo=50):  # Set headless to True by default
        self.headless = headless
        self.slow_mo = slow_mo
        self.playwright = None
        self.browser = None
        self.page = None
        
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        self.page = self.browser.new_page()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
            
    def login(self, email, password):
        """
        Handle the login process. You may need to adjust this based on ChatGPT's login page structure.
        May also need to handle 2FA if enabled.
        """
        self.page.goto('https://chat.openai.com/auth/login')
        
        # Click the login button (might change based on UI)
        self.page.click('button:has-text("Log in")')
        
        # Fill the email
        self.page.fill('input[name="username"]', email)
        self.page.click('button:has-text("Continue")')
        
        # Fill the password
        self.page.fill('input[name="password"]', password)
        self.page.click('button[type="submit"]')
        
        # Wait for login to complete
        try:
            self.page.wait_for_selector('textarea', timeout=10000)
            print("Login successful")
        except TimeoutError:
            print("Login failed or page structure changed")
            
    def load_from_existing_session(self, cookies_path=None):
        """
        Alternative to login - use an existing browser session
        You can save/load cookies to maintain session
        """
        if cookies_path:
            import json
            with open(cookies_path, 'r') as f:
                cookies = json.load(f)
            self.page.context.add_cookies(cookies)
        
        self.page.goto('https://chat.openai.com/')
        
    def simulate_realistic_typing(self, text, input_selector='textarea', error_probability=0.03):
        """
        Simulate realistic human typing with occasional typos and corrections
        """
        input_element = self.page.locator(input_selector)
        
        i = 0
        while i < len(text):
            # Random delay between keystrokes
            time.sleep(random.normalvariate(0.08, 0.02))  # Mean 80ms, SD 20ms
            
            # Occasional longer pause (thinking pause)
            if random.random() < 0.02:  # 2% chance
                time.sleep(random.uniform(0.5, 1.5))
            
            # Simulate occasional typing error
            if random.random() < error_probability and i < len(text) - 1:
                # Type a wrong character
                typo_char = chr(ord(text[i]) + random.randint(-2, 2))
                input_element.type(typo_char, delay=random.normalvariate(0.05, 0.01))
                
                # Pause briefly as if noticing the error
                time.sleep(random.uniform(0.1, 0.3))
                
                # Press backspace to correct
                self.page.keyboard.press('Backspace')
                
                # Don't increment i, so we type the correct character next
                continue
            
            # Type the correct character
            input_element.type(text[i], delay=random.normalvariate(0.05, 0.01))
            i += 1
            
            # Occasional brief pause in the middle of typing
            if random.random() < 0.1 and i < len(text) - 1:  # 10% chance
                time.sleep(random.uniform(0.05, 0.2))
    
    def send_message(self, message):
        """
        Send a message to ChatGPT and wait for the response
        """
        # Make sure we're on the right page
        if "chat.openai.com" not in self.page.url:
            self.page.goto('https://chat.openai.com/')
            
        # Focus on the chat interface
        try:
            # Find the input box and click on it
            input_box = self.page.locator('textarea')
            input_box.click()
            
            # Simulate realistic typing
            self.simulate_realistic_typing(message)
                
            # Send the message
            self.page.keyboard.press('Enter')
            
            # Wait for the response to complete
            self.wait_for_response_complete()
            
            # Get the latest response
            responses = self.page.locator('.markdown').all()
            
            if responses:
                latest_response = responses[-1].inner_text()
                return latest_response
            else:
                return "No response detected"
                
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
            
    def wait_for_response_complete(self, timeout=60000):
        """
        Wait for ChatGPT to finish generating its response
        """
        try:
            # First wait for the response to start
            self.page.wait_for_selector('.markdown', timeout=timeout)
            
            # Then wait for the stop generating button to disappear
            try:
                self.page.wait_for_selector('button:has-text("Stop generating")', timeout=5000)
                self.page.wait_for_selector('button:has-text("Stop generating")', state='hidden', timeout=timeout)
            except TimeoutError:
                # No stop generating button appeared, which might be fine for short responses
                pass
            
            # Add a small delay to ensure the response is fully rendered
            time.sleep(1)
            
        except TimeoutError:
            print("Timeout waiting for response")
            
    def start_new_conversation(self):
        """Start a new conversation"""
        self.page.goto('https://chat.openai.com/')
        try:
            # Click the "New chat" button
            self.page.click('button:has-text("New chat")')
            time.sleep(1)
        except:
            print("Could not start new conversation, may already be on a new chat")
            
    def process_bulk_queries(self, queries, save_results_to=None, new_chat_for_each=True):
        """
        Process a list of queries and optionally save results
        """
        results = []
        
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            if new_chat_for_each:
                self.start_new_conversation()
                
            # Add random delay between queries to seem more human
            if i > 0:
                time.sleep(random.uniform(3, 10))
                
            response = self.send_message(query)
            results.append({"query": query, "response": response})
            
            # Random pause between queries
            time.sleep(random.uniform(2, 5))
            
        # Save results if requested
        if save_results_to:
            import json
            with open(save_results_to, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        return results

# Example usage with headless mode
def run_example():
    # List of queries to process
    queries = [
        "Explain the concept of browser automation",
        "What are some ethical considerations when using automation tools?",
        "Give me a simple Python function to calculate Fibonacci numbers"
    ]
    
    with ChatGPTWebAutomation(headless=True) as bot:  # Explicitly set headless=True
        # Either login with credentials
        # bot.login("your_email@example.com", "your_password")
        
        # Or use existing session cookies (you'll need to create this file on a machine where you can log in)
        bot.load_from_existing_session(cookies_path="chatgpt_cookies.json")
        
        # Process all queries and save results
        results = bot.process_bulk_queries(
            queries,
            save_results_to="chatgpt_responses.json",
            new_chat_for_each=True
        )
        
        # Print results
        for i, res in enumerate(results):
            print(f"\nQuery {i+1}: {res['query'][:50]}...")
            print(f"Response: {res['response'][:100]}...")

if __name__ == "__main__":
    run_example()