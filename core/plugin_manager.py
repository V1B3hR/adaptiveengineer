"""
Plugin manager for loading, managing, and coordinating plugins.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from .plugin_base import PluginBase


logger = logging.getLogger('plugin_manager')


class PluginManager:
    """
    Manages the lifecycle and coordination of plugins.
    
    Provides:
    - Plugin registration and discovery
    - Plugin lifecycle management (load, enable, disable, unload)
    - State aggregation across plugins
    - Signal routing to appropriate plugins
    """
    
    def __init__(self):
        """Initialize the plugin manager."""
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_types: Dict[str, List[str]] = {}  # type -> list of plugin_ids
        self._node = None
        
    def register_plugin(self, plugin: PluginBase) -> bool:
        """
        Register a plugin with the manager.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registered successfully, False if plugin_id already exists
        """
        if plugin.plugin_id in self.plugins:
            logger.warning(f"Plugin {plugin.plugin_id} already registered")
            return False
        
        self.plugins[plugin.plugin_id] = plugin
        
        # Track by type
        plugin_type = plugin.get_plugin_type()
        if plugin_type not in self.plugin_types:
            self.plugin_types[plugin_type] = []
        self.plugin_types[plugin_type].append(plugin.plugin_id)
        
        logger.info(f"Registered plugin: {plugin.plugin_id} (type: {plugin_type})")
        return True
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_id: ID of plugin to unregister
            
        Returns:
            True if unregistered successfully, False if plugin not found
        """
        if plugin_id not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        plugin_type = plugin.get_plugin_type()
        
        # Remove from type tracking
        if plugin_type in self.plugin_types:
            self.plugin_types[plugin_type].remove(plugin_id)
            if not self.plugin_types[plugin_type]:
                del self.plugin_types[plugin_type]
        
        del self.plugins[plugin_id]
        logger.info(f"Unregistered plugin: {plugin_id}")
        return True
    
    def initialize_all(self, node: Any) -> None:
        """
        Initialize all registered plugins with the host node.
        
        Args:
            node: Host node reference
        """
        self._node = node
        for plugin in self.plugins.values():
            try:
                plugin.initialize(node)
                logger.info(f"Initialized plugin: {plugin.plugin_id}")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin.plugin_id}: {e}")
    
    def update_all(self, delta_time: float) -> None:
        """
        Update all enabled plugins.
        
        Args:
            delta_time: Time elapsed since last update
        """
        for plugin in self.plugins.values():
            if plugin.is_enabled():
                try:
                    plugin.update_state(delta_time)
                except Exception as e:
                    logger.error(f"Error updating plugin {plugin.plugin_id}: {e}")
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Get a plugin by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[PluginBase]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugin instances
        """
        plugin_ids = self.plugin_types.get(plugin_type, [])
        return [self.plugins[pid] for pid in plugin_ids if pid in self.plugins]
    
    def get_all_state(self) -> Dict[str, Dict[str, float]]:
        """
        Get state from all plugins.
        
        Returns:
            Dictionary mapping plugin_id to state dictionary
        """
        return {
            plugin_id: plugin.get_state()
            for plugin_id, plugin in self.plugins.items()
            if plugin.is_enabled()
        }
    
    def route_signal(self, signal: Any) -> List[Any]:
        """
        Route a signal to all relevant plugins.
        
        Args:
            signal: Signal to route
            
        Returns:
            List of responses from plugins
        """
        responses = []
        for plugin in self.plugins.values():
            if plugin.is_enabled():
                try:
                    response = plugin.process_signal(signal)
                    if response is not None:
                        responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing signal in plugin {plugin.plugin_id}: {e}")
        return responses
    
    def execute_action(self, plugin_id: str, action_type: str, params: Dict[str, Any]) -> bool:
        """
        Execute an action on a specific plugin.
        
        Args:
            plugin_id: Target plugin ID
            action_type: Action to execute
            params: Action parameters
            
        Returns:
            True if action executed successfully
        """
        plugin = self.get_plugin(plugin_id)
        if plugin and plugin.is_enabled():
            return plugin.execute_action(action_type, params)
        return False
    
    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered plugins.
        
        Returns:
            Dictionary with plugin information
        """
        return {
            plugin_id: {
                'type': plugin.get_plugin_type(),
                'enabled': plugin.is_enabled(),
                'state_variables': list(plugin.get_state_schema().keys()),
                'actions': plugin.get_actions()
            }
            for plugin_id, plugin in self.plugins.items()
        }
