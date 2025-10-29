# General rules
- We build new system, so we never need to migrate our data, we never write fallback code
- Do not create any examples
- Do not create any tests if you are not asked to do it - usualy the user test the code by his own
- Do not write class or method comments

- Never use emoticons in log messages
- Never use step numbers or progress indicators in log messages (like "Step 1/3", "Phase 2", etc.)
- Log messages should be focused on application domain, never on your reasoning or conversation
- Fail fast, raise exceptions
- if accessing data from dictionary variable, assume data is there, dont provide default values if not asked