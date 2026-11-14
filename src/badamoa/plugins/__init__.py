"""Scripts that run inside plugin environments.

They are started by file path with the environment's own Python, so they must
not import ``badamoa`` — only the standard library and the plugin's requirements.
They talk to the node through stdout: ordinary lines are the job log, and lines
starting with ``@@badamoa `` carry a JSON event (``info``, ``epoch``, ``result``).
"""
