import csv
import itertools
import sys

PROBS = {                                                                       # dictionary containing a number of constants representing probabilities of various different events. 

    # Unconditional probabilities for having gene
    "gene": {                                                                   # the probability if we know nothing about that person’s parents.
        2: 0.01,                                                                # there’s a 1% chance of having 2 copies of the gene
        1: 0.03,                                                                # a 3% chance of having 1 copy of the gene
        0: 0.96                                                                 # a 96% chance of having 0 copies of the gene.
    },

    "trait": {                                                                  # represents the conditional probability that a person exhibits a trait (like hearing impairment). 

        # Probability of trait given two copies of gene
        2: {                                                                    # the probability distribution that a person has the trait given that they have two versions of the gene
            True: 0.65,                                                         # they have a 65% chance of exhibiting the trait
            False: 0.35                                                         # and a 35% chance of not exhibiting the trait.
        },

        # Probability of trait given one copy of gene
        1: {                                                                    # if a person has 1 copies of the gene,
            True: 0.56,                                                         # they have a 56% chance of exhibiting the trait
            False: 0.44                                                         # and a 44% chance of not exhibiting the trait.
        },

        # Probability of trait given no gene
        0: {                                                                    # if a person has 0 copies of the gene,
            True: 0.01,                                                         # they have a 1% chance of exhibiting the trait
            False: 0.99                                                         # and a 99% chance of not exhibiting the trait.
        }
    },

    # Mutation probability
    "mutation": 0.01                                                            #  If a mother has two versions of the gene, for example, and therefore passes one on to her child, there’s a 1% chance it mutates into not being the target gene anymore.
                                                                                # Conversely, if a mother has no versions of the gene, and therefore does not pass it onto her child, there’s a 1% chance it mutates into being the target gene
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])                                             # loads data from a file into a dictionary people.

    # Keep track of gene and trait probabilities for each person
    probabilities = {                                                           # defines a dictionary of probabilities
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):                                          # Who might have the trait (powerset(names)).

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):                                        # Who might have 1 gene.
            for two_genes in powerset(names - one_gene):                        # Who might have 2 genes.

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)  # For each scenario → compute joint probability → update the distributions.
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:                                                   # Returns a dictionary like: {
        reader = csv.DictReader(f)                                              # "Harry": {"mother": "Lily", "father": "James", "trait": True},
        for row in reader:                                                      # "James": {"mother": None, "father": None, "trait": False},
            name = row["name"]                                                  # "Lily": {"mother": None, "father": None, "trait": True}
            data[name] = {                                                      # }
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):                                                                # Used for testing all combinations of who might have 1 gene, 2 genes, or the trait.
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probability = float(1)

    for person in people:
        genes = (                                                               # Figure out how many genes they have (0, 1, or 2).
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )

        trait = person in have_trait
        mother = people[person]["mother"]
        father = people[person]["father"]

        if mother is None and father is None:                                   # If they have no parents listed → use unconditional probability from PROBS["gene"].
            probability *= PROBS["gene"][genes]

        else:                                                                   # If they have parents → compute probability of inheriting gene from mom/dad:
            passing = {mother: 0, father: 0}

            for parent in passing:
                passing[parent] = (
                    1 - PROBS["mutation"] if parent in two_genes else           # Parent with 2 genes → passes gene with prob 0.99.
                    0.5 if parent in one_gene else                              # Parent with 1 gene → passes with prob 0.5.
                    PROBS["mutation"]                                           # Parent with 0 genes → passes with prob 0.01 (mutation).
                )

            probability *= (                                                    # Then combine mother + father to get child’s gene probability.
                passing[mother] * passing[father] if genes == 2 else
                passing[mother] * (1 - passing[father]) + (1 - passing[mother]) * passing[father] if genes == 1 else
                (1 - passing[mother]) * (1 - passing[father])
            )
        probability *= PROBS["trait"][genes][trait]                             # Multiply by probability of trait given that gene count: PROBS["trait"][genes][trait].

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        genes = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )

        trait = person in have_trait
                                                                                # Example: if p = 0.0023 and Harry had 1 gene and trait:
        probabilities[person]["gene"][genes] += p                               # probabilities["Harry"]["gene"][1] += 0.0023
        probabilities[person]["trait"][trait] += p                              # probabilities["Harry"]["trait"][True] += 0.0023


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:                                                # It divides each value by the total to make them sum to 1.
        gene_sum = 0                                                            # Example before normalization:
        for i in range(3):                                                      # Harry gene: {0: 0.23, 1: 0.34, 2: 0.43} (sum = 1.00 ✅)
            gene_sum += probabilities[person]["gene"][i]                        # James gene: {0: 0.5, 1: 0.2, 2: 0.1} (sum = 0.8 ❌)

        for i in range(3):
            probabilities[person]["gene"][i] /= gene_sum

        trait_sum = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        probabilities[person]["trait"][True]  /= trait_sum
        probabilities[person]["trait"][False] /= trait_sum


if __name__ == "__main__":
    main()
