# alpha-gomoku-zero

- adopts similar architecture to AlphaGo Zero
- used for Gomoku (Five in a Row) game
- a 10x10 board setup
- an optional pre-training phase included (so not that zero)
- trainable but not yet learned
- plausible but not yet intelligent
- a work in progress
- only for reference

## futher directions

- when mcst should be deeper and when should be shallower?
- add more later-staged samples (either self-play or supervised pre-training)
    - poorer performance at the later stage is found by other students, probably beneficial
    - sparse reward is a problem especially at the early stage, but not later
    - similar to subgames in game theory
- only attacks but not defends
