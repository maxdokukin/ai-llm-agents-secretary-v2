ContextManager
A server to manage the model context window
stores:
system prompt
tools (with reason added, id_added)
relevant data (with reason added, id_added)
message history (id_added)
cache index

There is a dynamic garbage collector/caching mechanism moves not so relevant data in the cache while reserving an indexed entry in the context
Caching: stores older information in the cache rather than in the main context
things that can be cached: tools
every cache entry has a hit rate based on which the manager dynamically loads/unload the data/tool/message from the history

LLMServer
Starts and maintains the local LLM Server using llama cpp
Has options for Gemma 4 models pulled from the HF

ToolManager
Defines and stores all the tools available to the LLM
there is select_best_tool api that returns the name of the most suitable tool for a given problem statement
tool index is exposed in the context, but tools descriptions are only loaded on demand, stored, and cached when unused

StatusManager
Dynamic status of the current LLM phase