# SatQuery AI — Setup Guide

## Purpose

This guide explains how to set up the SatQuery AI project locally for development, testing, and evaluation.

The setup process includes creating a Python virtual environment, installing the required dependencies, and running the evaluation tests.

## Prerequisites

Before setting up the project, make sure the following are installed:

- Python 3.11 or compatible Python 3 version
- Git
- Visual Studio Code (recommended)
- Internet connection for installing Python packages

The evaluation environment has been tested with Python 3.11.6.

## Clone the Repository

Clone the SatQuery AI repository using Git:

```bash
git clone <https://github.com/thodetivarshith/satcoders-satquery-ai.git>
cd satcoders-satquery-ai

## Create a Virtual Environment

Create a Python virtual environment for the project:

```bash
python -m venv .venv

## Install Dependencies

Activate the virtual environment and install the required Python packages:

```bash
pip install numpy pandas scikit-learn pytest matplotlib requests

## Run Evaluation Tests

From the project root directory, run the evaluation test suite:

```bash
python -m pytest evaluation\test_data.py

## Run the Benchmark

From the project root directory, run:

```bash
python -m evaluation.benchmark

## Evaluation Project Structure

The main evaluation-related files are organized as follows:

```text
evaluation/
├── datasets/
├── results/
├── benchmark.py
├── metrics.py
├── README.md
└── test_data.py