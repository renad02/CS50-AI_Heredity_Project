# CS50's AI Heredity Project

This project builds a model that infers the likelihood of individuals carrying a particular gene and expressing a related trait, using a Bayesian network and inference by enumeration. It’s drawn from CS50’s Introduction to Artificial Intelligence with Python

**Key Concepts:**
* Bayesian Network Structure
  * Each person has two random variables:
    * Gene: Number of copies (0, 1, or 2) of the target gene.
    * Trait: Whether the trait is expressed, influenced by the number of gene copies.
  * Dependencies:
    * The Gene of each person depends on their parents’ Genes.
    * The Trait depends on that individual’s Gene.
* Probabilistic Modeling (PROBS)
  * PROBS["gene"]: Unconditional probabilities of having 0, 1, or 2 gene copies (e.g., 0 → 96%, 1 → 3%, 2 → 1%)
  * PROBS["trait"]: Likelihood of expressing the trait based on gene count (e.g., with 2 genes, a 65% chance of trait expression)
  * PROBS["mutation"]: Mutation rate (1%) — the chance a gene copy flips when passed from parent to child

**Project Goals:**

Implement three core functions:

1- joint_probability(people, one_gene, two_genes, have_trait)
  * Calculates the joint probability that each person has a specified gene count and exhibits (or doesn’t exhibit) the trait, given:
    * people: dataset mapping each person to parent info and observed trait (True/False/None).
    * one_gene: set of people with exactly 1 gene copy.
    * two_genes: set of people with exactly 2 gene copies.
    * have_trait: set of people expressing the trait.
  * For unobserved individuals, defaults to 0 gene copies or no trait, accordingly.
  * Uses:
    * Unconditional gene probabilities for those without parent data.
    * Inheritance modeling with mutation for those with parent info: each parent passes a gene copy—possibly mutating with probability PROBS["mutation"].
    * Trait expression probability based on gene count using PROBS["trait"].

2- update(probabilities, one_gene, two_genes, have_trait, p)
  * Adds the computed joint probability p into aggregate probability distributions:
    * probabilities[person]["gene"][i] where i is 0, 1, or 2.
    * probabilities[person]["trait"][True or False].
  * Does not return a value—updates data in place.

3- normalize(probabilities)
  * Adjusts all probability distributions so they each sum to 1, preserving relative weights.

**Workflow Summary:**
  * Load data from a CSV mapping individuals to parents and trait observations.
  * Initialize probability structures for each person (gene and trait distributions starting at 0).
  * Iterate over all possible combinations of:
    * Who might have 0, 1, or 2 gene copies.
    * Who might exhibit the trait.
  * For each combination:
    1- Compute its joint probability.
    2- Use update(...) to contribute that to the overall distributions.
  * After processing all combinations, call normalize(...) to produce valid distributions.

**Why It Matters ?**

This project demonstrates:
  * Bayesian reasoning with uncertain, partially observed data.
  * Inference by enumeration, exploring all possible configurations.
  * Application of probabilistic modeling, including mutation and inheritance mechanics.
  * Careful design of data structures and clean modular functions—key in real-world AI systems.
