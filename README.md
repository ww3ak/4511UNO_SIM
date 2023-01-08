# Tackling the UNO Card Game with Reinforcement Learning

Author: [Bernhard Pfann](https://www.linkedin.com/in/bernhard-pfann/)<br>
Status: Work in progress

## Description
In this project I tried to analytically derive an optimal strategy, for the classic UNO card game. To do so, I structured my work as follows:
1. Creating a game engine of the UNO card game in Python from scratch
2. Obtaining game statistics from simulating a series of 100,000 games
3. Implementing basic Reinforcement Learning techniques (Q-Learning & Monte Carlo) in order to discover an optimal game strategy

<b>UNO card engine:</b> In order to train a Reinforcement Learning (RL) agent how to play intelligently, a fully-fledged game environment needs to be in place, capturing all the mechanics and rules of the game. Class objects for <code>Card</code>, <code>Deck</code>, <code>Player</code>, <code>Turn</code> and <code>Game</code> are defined.

<b>Statistics from simulations:</b> By running multiple simulations of the game, the following questions are being tackled:
* How many turns do games last?
* How big is the advantage of the player making the first turn?
* What are the most likely situations in the course of a game?

<b>Application of Reinforcement Learning:</b> In agent.py, I defined the algorithm for a Q-Learning or Monte-Carlo agent, both working with a discrete state-action matrix. In `config.py` the preferred algorithm can be selected together with its main tuning parameters. Finally the results after training the RL-model are being analyzed in terms to cumulative win-rate and obtained Q-values.

## Repository Structure

 - `assets/` collection of .csv files that are generated by execution of `run.py`
 - `notebooks/` analysis of simulated games
 - `src/` core package to simulate games
 - `config.py` configurable parameters
 - `run.py` executation file

## Execution Instruction

Clone repository via HTTPS:

```bash
$ git clone https://github.com/bernhard-pfann/uno-card-game-rl.git
```

Install requirements and execute main file:

```bash
$ pip install -r requirements.txt
$ python run.py
```
