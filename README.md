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

| Duration | Result |
|---------|--------|
| **1200 sec** | ![](assets/Screenshot%202026-04-27%20142041.png) |
| **600 sec**  | ![](assets/Screenshot%202026-04-27%20142250.png) |
| **300 sec**  | ![](assets/Screenshot%202026-04-27%20142400.png) |

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

| Metric | Value |
|-------|------|
| 🚗 Vehicles Processed | 576 |
| ⏱️ Avg Time per Vehicle | 33 Seconds (Reduced 50%) |
| ⚖️ Lane Balance | Highly Balanced |


---
## 🏗️ Project Architecture

TCS/
├── rl/
│   ├── traffic_env.py     # RL Environment (SUMO + Gym)
│   ├── train.py           # PPO Training Script
│   └── test_model.py      # Testing with GUI
│
├── models/
│   ├── ppo_22000.zip      # Final trained model
│   └── vec_normalize.pkl  # Normalization stats
│
├── config/
│   └── simulation.sumocfg # SUMO simulation config
│
├── network/
│   ├── city.net.xml       # Road network
│   └── routes.rou.xml     # Traffic routes
│
├── assets/
│   ├── 1200_sec.png
│   ├── 600_sec.png
│   └── 300_sec.png
│
├── main.py                # Final simulation runner
├── README.md              # Project documentation
└── requirements.txt       # Dependencies

---

## ⚙️ Installation & Usage

### 1. Install Dependencies
pip install stable-baselines3 gymnasium

### 2. Train the AI Agent
python rl/train.py

### 3. Run Simulation with GUI
python rl/test_model.py

---

## 🧪 Technical Deep Dive

State Space: [Waiting Time, Queue Length, Vehicle Count, Current Phase]  
Action Space: Discrete(4) (Selecting the optimal traffic phase)

Reward Function:  
Reward=−(0.05⋅W)−(0.5⋅Q)+(10.0⋅P)

(Where $W$=Wait time, $Q$=Queue, $P$=Passed vehicles)

---

## 🔮 Future Roadmap

[1] Multi-Agent RL: Controlling multiple intersections simultaneously.

[2] Yellow Phase: Adding transition safety periods.

[3] Dashboard: Web-based analytics for traffic engineers.

---

## 👨‍💻 Author

Bhanu  
Full-Stack Developer & AI Enthusiast

<p align="center"> 
<a href="https://www.google.com/search?q=https://github.com/Bhanubasyan"> 
<img src="https://www.google.com/search?q=https://img.shields.io/badge/GitHub-Profile-181717%3Fstyle%3Dflat%26logo%3Dgithub"/> 
</a> 
</p>

<p align="center">🚀 Built with passion for AI & Smart Cities</p>