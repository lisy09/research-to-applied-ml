# Amazon Search: The Joy of Ranking Products (SIGIR’16)

> Amazon Search: The Joy of Ranking Products ([Blog](https://www.amazon.science/publications/amazon-search-the-joy-of-ranking-products), [Paper (SIGIR ’16)](https://assets.amazon.science/89/cd/34289f1f4d25b5857d776bdf04d5/amazon-search-the-joy-of-ranking-products.pdf), [Video](https://www.youtube.com/watch?v=NLrhmn-EZ88), [Code](https://github.com/dariasor/TreeExtra)) `Amazon`

## Paper Reading

Describe a general machine learning
framework used for ranking within categories, blending separate rankings in All Product Search, NLP techniques used
for matching queries and products, and algorithms targeted
at unique tasks of specific categories — books and fashion.

Choice of model for ranking: `Gradient Boosted Trees`.

Treat each query as a noun phrase and consider the
head of the noun phrase to be the core product type and
all other words in the query to be modifiers.

Bagged Trees need to be normalized, so do the features.
https://github.com/dariasor/TreeExtra