"""Support required by the Carbonite extension loader"""

import omni.ext


class _PublicExtension(omni.ext.IExt):
    """Object that tracks the lifetime of the Python part of the extension loading"""

    def on_startup(self):
        """Set up initial conditions for the Python part of the extension"""

    def on_shutdown(self):
        """Shutting down this part of the extension prepares it for hot reload"""
