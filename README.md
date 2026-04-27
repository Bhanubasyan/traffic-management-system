<h1 align="center">🚦 GreenFlow AI: Smart Traffic Control</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/RL-PPO-green?style=for-the-badge&logo=openai"/>
  <img src="https://img.shields.io/badge/SUMO-Simulation-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Optimized-success?style=for-the-badge"/>
</p>

<p align="center">
  <b>Deep Reinforcement Learning (PPO) integrated with SUMO for dynamic traffic signal optimization.</b>
</p>

---

## 🧠 Project Overview
> This system replaces traditional fixed-timer signals with an **AI Agent** that observes real-time traffic density and waiting times to make intelligent switching decisions.

<details>
<summary><b>🔍 What the Agent Learns (Click to expand)</b></summary>
<br>
<ul>
  <li>🚗 <b>Congestion Reduction:</b> Prioritizing lanes with higher vehicle density.</li>
  <li>⏱️ <b>Wait Time Minimization:</b> Lowering the average delay per vehicle.</li>
  <li>⚖️ <b>Fairness:</b> Ensuring no lane is left on red for too long.</li>
  <li>📉 <b>Emission Control:</b> Reducing idle time to lower carbon footprint.</li>
</ul>
</details>

---

## 🖼️ Simulation Gallery

| 🚦 Active Simulation | 🚗 Traffic Flow Analysis |
|:---:|:---:|
| ![Sim1](./assests/Screenshot%202026-04-18%20154009.png) | ![Sim2](./assests/Screenshot%202026-04-18%20154131.png) |

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Adaptive Control** | Dynamic signal switching based on live queue lengths. |
| **PPO Algorithm** | Uses Proximal Policy Optimization for stable and reliable learning. |
| **Smart Constraints** | Balanced Min (10s) and Max (90s) green time logic. |
| **Real-time Metrics** | Live monitoring of average waiting times and vehicle counts. |

---

## 📊 Benchmark Results (Final Test)

```text
┌────────────────────────────────────────────────────────┐
│  METRIC                    │         VALUE             │
├────────────────────────────┼───────────────────────────┤
│ 🚗 Vehicles Processed      │  576                      │
│ ⏱️ Avg Time per Vehicle    │  33 Seconds (Reduced 50%) │
│ ⚖️ Lane Balance       │  Highly Balanced       │
└────────────────────────────────────────────────────────┘

 ---
##  Project Architecture

TCS/
├── rl/
│   ├── traffic_env.py   # RL Environment Wrapper
│   ├── train.py         # Training Script
│   └── test_model.py    # GUI Testing Script
├── models/              # Saved PPO Models (.zip)
├── config/              # SUMO Configuration Files
└── network/             # Map and Road Network Files

---

## Installation & Usage

## 1. Install Dependencies
pip install stable-baselines3 gymnasium

## 2. Train the AI Agent
python rl/train.py

## 3. Run Simulation with GUI
python rl/test_model.py

---

## 🧪 Technical Deep Dive
State Space: [Waiting Time, Queue Length, Vehicle Count, Current Phase]Action Space: Discrete(4) (Selecting the optimal traffic phase)Reward Function:$$Reward = -(0.05 \cdot W) - (0.5 \cdot Q) + (10.0 \cdot P)$$(Where $W$=Wait time, $Q$=Queue, $P$=Passed vehicles)

---

## 🔮 Future Roadmap
[ ] Multi-Agent RL: Controlling multiple intersections simultaneously.

[ ] Yellow Phase: Adding transition safety periods.

[ ] Dashboard: Web-based analytics for traffic engineers.


---

## 👨‍💻 Author
Bhanu Basyan
Full-Stack Developer & AI Enthusiast

<p align="center">
<a href="https://www.google.com/search?q=https://github.com/your-username">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/GitHub-Profile-181717%3Fstyle%3Dflat%26logo%3Dgithub"/>
</a>
</p>

<p align="center">🚀 Built with passion for AI & Smart Cities</p>