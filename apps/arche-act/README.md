

# ArcheAct: An LLM-Based Multi-Agent Framework with Disambiguation and Polymorphic Role-Playing

<div align="center">

  <a>![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)</a>
</div>

<p align="center">
  <img src='misc/archeact_framework.png' width=600>
</p>

## Overview

To address the challenges of automating data annotation for complex problem-solving, we introduce a novel multi-agent framework called ArcheAct. 
This framework is built upon conversational LLMs and incorporates a disambiguation mechanism and a polymorphic role-playing approach. 
The disambiguation mechanism ensures that user queries are precise and actionable within the task's context. 
Moreover, through multi-step reasoning facilitated by communicative agents and implementing a polymorphic role-playing strategy, ArcheAct can navigate the intricacies of real-world tasks that are too complex for a single agent to handle. 

## Try it yourself
We provide a demo showcasing the application of ArcheAct to simulate the travel assistant multi-agent system for searching POIs, navigating, obtaining travel advice, etc.

<p align="center">
  <img src='./misc/demo_showcase.gif' width=800>
</p>



### Installation

Install `ArcheAct` from source with conda and pip: 

```sh
# Create a conda virtual environment
conda create --name arche python=3.10

# Activate arche conda environment
conda activate arche

# Clone github repo
git clone https://github.com/PaddlePaddle/PaddleSpatial

# Change directory into project directory
cd PaddleSpatial/apps/arche-act

# Install ArcheAct from source
pip install -e .
```

### Example


**For Bash shell (Linux, macOS, Git Bash on Windows):**

```bash
# Export your ERNIE Bot Access Token
export ERNIE_BOT_ACCESS_TOKEN=<insert your EB Access Token>
```

**For Windows Command Prompt:**

```cmd
REM export your ErnieBot Access Token
set ERNIE_BOT_ACCESS_TOKEN=<insert your ErnieBot Access Token>
```

**For Windows PowerShell:**

```powershell
# Export your OpenAI API key
$env:ERNIE_BOT_ACCESS_TOKEN="<insert your ErnieBot Access Token>"
```

Replace `<insert your ErnieBot Access Token>` with your actual ErnieBot Access Token in each case. 
Make sure there are no spaces around the `=` sign.

After setting the API key, you can run the script:

```bash
# A web-based user interface demo
python apps/agents/agents_mixact.py
```

Please note that the environment variable is session-specific. 
If you open a new terminal window or tab, you will need to set the access token again in that new session.



## License

The intended purpose and licensing of ArcheAct is solely for research use.

The source code is licensed under Apache 2.0.

