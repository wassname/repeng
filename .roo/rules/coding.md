
## Code

When writing code, first understand the problem, and present a plan to the user in the form of a markdown checklist. Then if you get approval to act, maintain a markdown checklist to keep track of tasks completions e.g.

- [x] Task 1
- [/] Task 2
- [ ] Task 3

When editing code or words we want it to get better at every step, so the information content should increase, and the cost to decode/grok should decrease. i.e. it should be more organised, clearer to the audience, and more correct. Decrease etropy for the user. 

If you are unsure how to do that, cultivate the scout mindset and ask questions until you are certain. Your edits sizes should be proportional to your certainty about the content and it's context. If you are not sure, make small edits and ask questions and sanity checks. I know I'm not always the best writer or organiser so I appreciate help.

## Python Coding Style

In python or markdown. When using equations, make a markdown block and define the equations in latex along with variables, units, assumptions, name/premise of equation, etc, any any sources

Where useful, use FIXME, TODO, HACK to explicitly mark corresponding code

In python, if you need to document you can use function or variable names, types (for self documenting) and/or docstrings ruff/pep8

when relevant use var names to be clear about arrays vs scalars and normalised vs unnormalised
I occasionally use the libraries: dataset, anycache, lightning,
when coding math I like define terms as greek letters, symbols so the math is coded in a few lines of concise math
Use loguru not print or logging
No massive objects if they don't use self or don't help organise the code. No deep inheritance chains - composition over inheritance

I like my comments to concisly communicate intent and things that are not obvious from the code. So when I need to explain something about my code to someone, it's a sign it's not self-evident and I need to refactor or document or comment to make it clearer. This keeps it concise while explaining what code doesn't show directly (intent, why certain choices).


# Overall coding style

Educational Clarity:

    Every file is readable top-to-bottom
    Inline comments explain "why" not just "what"
    No hidden magic or auto-configuration
    Logging shows exactly what's happening

Anti-Framework Philosophy (for research code):

    No if-then-else monsters based on model type
    Single "strong baseline" instead of "flexible framework"
