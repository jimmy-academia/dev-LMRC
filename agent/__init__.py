"""
Agent package initialization file.
Provides the get_agent function for importing and initializing different agent types.
"""

def get_agent(agent_type):
    """
    Factory function to get the appropriate agent class based on the type.
    
    Args:
        agent_type (str): Type of agent to use ('base', 'recursive', 'bag')
        
    Returns:
        The agent class (not an instance)
    """
    if agent_type == 'recursive':
        from .recursive import RecursiveAgent
        return RecursiveAgent
    elif agent_type == 'bag':
        from .bag import BagAgent
        return BagAgent
    elif agent_type == 'base':
        from .base import ReactAgent
        return ReactAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")