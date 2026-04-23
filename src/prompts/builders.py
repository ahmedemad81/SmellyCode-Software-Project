# src/prompts/builders.py

from src.constants import LABELS


def build_classifier_input(code: str) -> str:
    """
    Build structured input for sequence classification.
    Optimized to reduce confusion between feature_envy and long_method.
    """

    return f"""You are a software quality expert.

Task:
Classify the following Java code into exactly ONE code smell category.

Available labels:
- {LABELS[0]}
- {LABELS[1]}
- {LABELS[2]}
- {LABELS[3]}

Definitions:

- data_class:
  Class mainly contains fields with simple getters/setters and very little logic.

- feature_envy:
  A method interacts more with another class than its own.
  Look for excessive calls to external objects or use of foreign data.

- god_class:
  A class that is too large and handles many responsibilities.
  Often contains many methods and diverse logic.

- long_method:
  A single method that is very long or complex.
  Focus on size, nesting, and number of operations inside ONE method.

Important distinctions:

- feature_envy is about **cross-class interaction**
- long_method is about **method size and complexity**
- Do NOT confuse long code with feature_envy unless it heavily depends on other classes

Java Code:
{code}
"""