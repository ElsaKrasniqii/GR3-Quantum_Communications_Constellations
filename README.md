#  Quantum Communications Constellations – Genetic Algorithm 


##  Authors
**Elsa Krasniqi**  

**Elona Kuqi** 

**Fahrije Gjokiqi** 

**Florjete Kuka** 

- Faculty of Electrical and Computer Engineering
- Computer and Software Engineering
- Subject: **Design and Analysis of Algorithms**
- Year: 2025



## **Overview**

This project is part of **SpOC 2: Quantum Communications Constellations**, a space challenge organized by **ESA's Advanced Concepts Team** and hosted at **GECCO 2023**. 

The goal of this project is to optimize the design of **constellations** for quantum communication between **rovers** and **motherships** in a Mars mission. This optimization needs to balance **communication efficiency** and the **operational costs** of satellite and rover communication.

### **Problem Description:**
- Engineers are tasked with finding optimal configurations for reliable communication between the Mars surface and satellites in orbit.
- The challenge involves determining the **number of satellites**, their **orbital parameters**, and their **positioning** for two Walker Delta Pattern constellations.

---

## **Objectives**

This project has two primary objectives:
1. **Minimizing the average communication cost** between the 7 motherships and the 4 rovers via the Walker constellation satellites.
2. **Minimizing the total cost** of manufacturing, launching, and operating the two constellations. This involves considering the number of satellites and their quality.

---

## **Encoding of Decisions and Parameterization**

This project uses a **decision vector** consisting of parameters for orbital configurations and rover positioning:
1. **Orbital parameters** of the two Walker constellations: semi-major axis (a), eccentricity (e), inclination (i), argument of perigee (w), and quality indicator (η).
2. **Parameters** for the number of satellites, planes, and phasing for each constellation.
3. **Positioning of rovers** on the surface, encoded as indices of possible rover positions in a **file** (`rovers.txt`).

Each decision vector **x** is represented as:

```plaintext
x = [a1, e1, i1, w1, eta1] + [a2, e2, i2, w2, eta2] + [S1, P1, F1] + [S2, P2, F2] + [r1, r2, r3, r4]

Using Data Files
Rover Data File (rovers.txt)

## **Rover Data File (`rovers.txt`)**
This file contains 100 possible positions for the rovers, formatted as

Lat (rad.)   Lon (rad.)
3.290891754147420301e-01  3.455813102533720649e+00
9.210033238890864560e-01  1.222549040618014837e+00
...

