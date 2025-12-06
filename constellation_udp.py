import pykep as pk
import numpy as np
import scipy
import os
import networkx as nx
from matplotlib import pyplot as plt
from sgp4.api import Satrec, SatrecArray, WGS72
import warnings
warnings.filterwarnings("ignore")


###############################################################
# MOTHERSHIP TLEs
###############################################################
def get_mothership_satellites():
    tles = [
        [
            "1 39634U 14016A   22349.82483685  .00000056  00000-0  21508-4 0  9992",
            "2 39634  98.1813 354.7934 0001199  83.3324 276.7993 14.59201191463475"
        ],
        [
            "1 26400U 00037A   00208.84261022 +.00077745 +00000-0 +00000-0 0  9995",
            "2 26400 051.5790 297.6001 0012791 171.3037 188.7763 15.69818870002328"
        ],
        [
            "1 36508U 10013A   22349.92638064  .00000262  00000-0  64328-4 0  9992",
            "2 36508  92.0240 328.0627 0004726  21.3451 338.7953 14.51905975672463"
        ]
    ]
    return [Satrec.twoline2rv(t[0], t[1]) for t in tles]


###############################################################
# CONSTELLATION UDP — FINAL QKD VERSION A (MODIFIED FOR NSGA-II)
###############################################################
class constellation_udp:

    def __init__(self):
        # TIME GRID
        self._t0 = 10000
        self.n_epochs = 11
        self._duration = 5

        jd0 = pk.epoch(self._t0, "mjd2000").jd
        self.jds = np.linspace(jd0, jd0 + self._duration * 365.25, self.n_epochs)
        self.frs = np.zeros_like(self.jds)

        # Load satellites
        mothers = get_mothership_satellites()
        self.pos_m = self._propagate_mothers(SatrecArray(mothers))
        self.n_motherships = len(mothers)

        # Load rover grid
        rover_file = "./data/spoc2/constellations/rovers.txt"
        if not os.path.exists(rover_file):
            # Create a default grid if file doesn't exist
            print(f"Warning: {rover_file} not found. Using default rover positions.")
            self.lambdas = np.array([0.1, 0.2, 0.3, 0.4])
            self.phis = np.array([0.0, 0.5, 1.0, 1.5])
        else:
            self.rovers_db = np.loadtxt(rover_file)
            self.lambdas = self.rovers_db[:, 0]
            self.phis = self.rovers_db[:, 1]
        
        self.n_rovers = 4


        # PHYSICAL CONSTANTS
        self.Rp = pk.EARTH_RADIUS / 1000
        self.LOS = 1.05 * self.Rp
        self.w_rot = 7.292115e-5
        self.eps_z = np.cos(np.radians(60))

        # HARD CONSTRAINT LIMITS
        self.min_rover_dist = 3000  # km
        self.min_sat_dist = 50      # km
        
        # PENALTY FACTORS FOR CONSTRAINT VIOLATIONS
        self.PENALTY_ROVER = 1000.0
        self.PENALTY_SAT = 1000.0


    ###############################################################
    # PYGMO REQUIRED METHODS (MODIFIED FOR NSGA-II)
    ###############################################################
    def get_nobj(self): 
        return 2  # Two objective functions
    
    def get_nec(self):  
        return 0  # No equality constraints (NSGA-II doesn't handle constraints)
    
    def get_nic(self):  
        return 0  # No inequality constraints (converted to penalty)


    def get_bounds(self):
        # ESA SPOC-2 VALID RANGES
        low = [
            6871, 0, 0, -np.pi, 0,           # a1, e1, i1, w1, eta1
            6871, 0, 0, -np.pi, 0,           # a2, e2, i2, w2, eta2
            1, 1, 0,                         # S1, P1, F1
            1, 1, 0,                         # S2, P2, F2
            0, 0, 0, 0                       # rover indices (4 values)
        ]
        high = [
            22371, 0.1, np.pi, np.pi, 200,   # a1, e1, i1, w1, eta1
            22371, 0.1, np.pi, np.pi, 200,   # a2, e2, i2, w2, eta2
            20, 20, 20,                      # S1, P1, F1
            20, 20, 20,                      # S2, P2, F2
            len(self.lambdas)-1,
            len(self.lambdas)-1,
            len(self.lambdas)-1,
            len(self.lambdas)-1
        ]
        return (low, high)


    ###############################################################
    # POSITION MODELS
    ###############################################################
    def _propagate_mothers(self, arr):
        err, pos, _ = arr.sgp4(self.jds, self.frs)
        return pos

    def rover_positions(self, lamb, phi):
        pos = np.zeros((self.n_rovers, self.n_epochs, 3))
        dt = (self.jds - self.jds[0]) * 86400

        for k, t in enumerate(dt):
            pos[:, k, 0] = self.Rp * np.cos(lamb) * np.cos(phi + self.w_rot*t)
            pos[:, k, 1] = self.Rp * np.cos(lamb) * np.sin(phi + self.w_rot*t)
            pos[:, k, 2] = self.Rp * np.sin(lamb)

        return pos


    ###############################################################
    # WALKER GENERATION — VERSION A (Stable ESA Walker Delta)
    ###############################################################
    def generate_walker(self, S, P, F, a, e, inc, w, t0):
       
        S, P, F = int(S), int(P), int(F)
        sats = []

        a_m = a * 1000
        n = np.sqrt(pk.MU_EARTH / a_m**3) * 60

        for p in range(P):
            raan = 360 * p / P
            for s in range(S):
                M = 360*(s/S + p*F/(P*S))


                sat = Satrec()


                sat.sgp4init(
                    WGS72, "i",
                    1000 + p*S + s,
                    t0,
                    0, 0, 0,
                    e,
                    np.degrees(w),
                    np.degrees(inc),
                    M,
                    n,
                    raan
                )
                sats.append(sat)

        return SatrecArray(sats)


    ###############################################################
    # COMBINE ALL POSITIONS
    ###############################################################
    def construct_walkers(self, x):
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, *_ = x
        w1s = self.generate_walker(S1, P1, F1, a1, e1, i1, w1, self._t0)
        w2s = self.generate_walker(S2, P2, F2, a2, e2, i2, w2, self._t0)
        return w1s, w2s

    def construct_pos(self, w1, w2, rv):
        err1, p1, _ = w1.sgp4(self.jds, self.frs)
        err2, p2, _ = w2.sgp4(self.jds, self.frs)
        return np.concatenate([p1, p2, self.pos_m, rv], axis=0)


    ###############################################################
    # QKD METRICS
    ###############################################################
    def line_of_sight(self, r1, r2):
        d = np.linalg.norm(r2 - r1)
        if d == 0: 
            return np.inf
        u = (r2 - r1) / d
        h = np.dot(r1, u)
        return np.sqrt(max(np.linalg.norm(r1)**2 - h**2, 0))

    def zenith(self, src, dst):
        d = dst - src
        return np.dot(d, src) / (np.linalg.norm(src) * np.linalg.norm(d))

    def qkd_metric(self, idx, src, dst, cosz, eta):
        d = np.linalg.norm(dst - src)
        d = max(d, 1)
        ew = -np.log(max(eta, 1e-6)) + 2 * np.log(d)

        if idx <= self.n_rovers:
            if cosz >= self.eps_z:
                zen = np.pi/2 - np.arccos(np.clip(cosz, -1, 1))
                ew += 1 / max(np.sin(zen), 1e-3)
            else:
                ew = 0
        return ew, d


    ###############################################################
    # GRAPH BUILDING
    ###############################################################
    def build_graph(self, ep, pos, n1, eta):
        N = pos.shape[0]
        adj = np.zeros((N, N))
        dmin = np.inf

        for i in range(N):
            for j in range(i):
                los = self.line_of_sight(pos[i], pos[j])
                cosz = self.zenith(pos[i], pos[j])
                if los < self.LOS and cosz <= 0:
                    ew, d = self.qkd_metric(N-i, pos[i], pos[j], cosz, 
                                           eta[0] if j < n1 else eta[1])
                    adj[i, j] = adj[j, i] = ew
                    dmin = min(dmin, d)

        return nx.from_numpy_array(adj), adj, dmin


    ###############################################################
    # SHORTEST PATH
    ###############################################################
    def avg_shortest_path(self, G, nm, nr):
        N = len(G.nodes())
        total = 0
        pairs = nm * nr

        for r in range(nr):
            for m in range(nm):
                try:
                    total += nx.shortest_path_length(
                        G, N-nm-nr+m, N-nr+r, weight="weight"
                    )
                except:
                    total += 1e4
        return total / max(pairs, 1)


    ###############################################################
    # CONSTRAINT CALCULATIONS (FOR PENALTY)
    ###############################################################
    def _calculate_rover_distance(self, lamb, phi):
        """Calculate minimum distance between rovers (for penalty calculation)."""
        pts = np.zeros((self.n_rovers, 3))
        pts[:, 0] = self.Rp * np.cos(lamb) * np.cos(phi)
        pts[:, 1] = self.Rp * np.cos(lamb) * np.sin(phi)
        pts[:, 2] = self.Rp * np.sin(lamb)

        def ang(u, v):
            return pk.EARTH_RADIUS / 1000 * np.arccos(
                np.clip(np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v)), -1, 1)
            )

        d = scipy.spatial.distance.cdist(pts, pts, ang)
        np.fill_diagonal(d, np.inf)
        return np.min(d)

    def _calculate_satellite_distance(self, w1, w2):
        """Calculate minimum distance between satellites (for penalty calculation)."""
        err1, p1, _ = w1.sgp4(self.jds, self.frs)
        err2, p2, _ = w2.sgp4(self.jds, self.frs)
        
        all_sats_pos = np.concatenate([p1, p2], axis=0)
        dmin_all = np.inf
        
        for ep in range(self.n_epochs):
            pos_ep = all_sats_pos[:, ep, :]
            n_sats = len(pos_ep)
            for i in range(n_sats):
                for j in range(i+1, n_sats):
                    d = np.linalg.norm(pos_ep[i] - pos_ep[j])
                    dmin_all = min(dmin_all, d)
        
        return dmin_all


    ###############################################################
    # FITNESS FUNCTION (MODIFIED WITH PENALTY METHOD)
    ###############################################################
    def fitness(self, x):
        # Extract parameters
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, *rv_idx = x
        
        # Convert to integers
        S1, P1, S2, P2 = int(S1), int(P1), int(S2), int(P2)
        N1 = S1 * P1
        N2 = S2 * P2
        
        # Get rover positions
        idx = [int(r) % len(self.lambdas) for r in rv_idx[:4]]
        lamb = self.lambdas[idx]
        phi = self.phis[idx]
        rv = self.rover_positions(lamb, phi)
        
        # Construct constellations
        w1s, w2s = self.construct_walkers(x)
        
        # ============================================================
        # CALCULATE OBJECTIVES
        # ============================================================
        pos = self.construct_pos(w1s, w2s, rv)
        
        f1 = 0  # Average shortest path
        for ep in range(1, self.n_epochs):
            G, adj, _ = self.build_graph(ep, pos[:, ep, :], N1, (eta1, eta2))
            f1 += self.avg_shortest_path(G, self.n_motherships, self.n_rovers)
        
        f1 /= (self.n_epochs - 1)
        f2 = eta1 * N1 + eta2 * N2  # Total cost
        
        # Normalize objectives
        f1_normalized = f1 / 34
        f2_normalized = f2 / 100000
        
        # ============================================================
        # CALCULATE CONSTRAINT VIOLATIONS AND PENALTIES
        # ============================================================
        penalty = 0.0
        
        # 1. Rover distance constraint penalty
        rover_distance = self._calculate_rover_distance(lamb, phi)
        if rover_distance < self.min_rover_dist:
            rover_violation = self.min_rover_dist - rover_distance
            penalty += self.PENALTY_ROVER * rover_violation
        
        # 2. Satellite distance constraint penalty
        sat_distance = self._calculate_satellite_distance(w1s, w2s)
        if sat_distance < self.min_sat_dist:
            sat_violation = self.min_sat_dist - sat_distance
            penalty += self.PENALTY_SAT * sat_violation
        
        # ============================================================
        # RETURN OBJECTIVES WITH PENALTY
        # ============================================================
        return [f1_normalized + penalty, f2_normalized + penalty]


    ###############################################################
    # UTILITY METHODS FOR DEBUGGING AND ANALYSIS
    ###############################################################
    def analyze_solution(self, x):
        """Analyze a solution and return detailed information."""
        # Extract parameters
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, *rv_idx = x
        
        S1, P1, S2, P2 = int(S1), int(P1), int(S2), int(P2)
        idx = [int(r) % len(self.lambdas) for r in rv_idx[:4]]
        lamb = self.lambdas[idx]
        phi = self.phis[idx]
        
        # Construct constellations
        w1s, w2s = self.construct_walkers(x)
        
        # Calculate distances
        rover_distance = self._calculate_rover_distance(lamb, phi)
        sat_distance = self._calculate_satellite_distance(w1s, w2s)
        
        # Check constraint violations
        rover_violated = rover_distance < self.min_rover_dist
        sat_violated = sat_distance < self.min_sat_dist
        
        # Calculate fitness without penalty (for comparison)
        fitness_without_penalty = self._fitness_without_penalty(x)
        
        return {
            'walker1': {'S': S1, 'P': P1, 'F': F1, 'satellites': S1 * P1},
            'walker2': {'S': S2, 'P': P2, 'F': F2, 'satellites': S2 * P2},
            'rovers': {'indices': idx, 'lambdas': lamb, 'phis': phi},
            'distances': {
                'rover_distance': rover_distance,
                'satellite_distance': sat_distance,
                'min_rover_dist': self.min_rover_dist,
                'min_sat_dist': self.min_sat_dist
            },
            'constraints': {
                'rover_violated': rover_violated,
                'sat_violated': sat_violated,
                'rover_violation_amount': max(0, self.min_rover_dist - rover_distance),
                'sat_violation_amount': max(0, self.min_sat_dist - sat_distance)
            },
            'fitness': {
                'without_penalty': fitness_without_penalty,
                'with_penalty': self.fitness(x)
            }
        }
    
    def _fitness_without_penalty(self, x):
        """Calculate fitness without penalty (for analysis only)."""
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, *rv_idx = x
        
        S1, P1, S2, P2 = int(S1), int(P1), int(S2), int(P2)
        N1 = S1 * P1
        N2 = S2 * P2
        
        idx = [int(r) % len(self.lambdas) for r in rv_idx[:4]]
        lamb = self.lambdas[idx]
        phi = self.phis[idx]
        rv = self.rover_positions(lamb, phi)
        
        w1s, w2s = self.construct_walkers(x)
        pos = self.construct_pos(w1s, w2s, rv)
        
        f1 = 0
        for ep in range(1, self.n_epochs):
            G, adj, _ = self.build_graph(ep, pos[:, ep, :], N1, (eta1, eta2))
            f1 += self.avg_shortest_path(G, self.n_motherships, self.n_rovers)
        
        f1 /= (self.n_epochs - 1)
        f2 = eta1 * N1 + eta2 * N2
        
        return [f1 / 34, f2 / 100000]


    ###############################################################
    # EXAMPLE SOLUTION
    ###############################################################
    def example(self):
        return [
            7000, 0.001, 1.2, 0, 40,      # Walker 1: a, e, i, w, eta
            8200, 0.001, 1.2, 0, 30,      # Walker 2: a, e, i, w, eta
            10, 2, 1,                     # S1, P1, F1
            8, 3, 1,                      # S2, P2, F2
            5, 10, 15, 20                 # Rover indices
        ]


    ###############################################################
    # PLOTTING
    ###############################################################
    def plot(self, x, ep=1):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        w1, w2 = self.construct_walkers(x)
        rid = [int(r) % len(self.lambdas) for r in x[-4:]]
        rv = self.rover_positions(self.lambdas[rid], self.phis[rid])
        pos = self.construct_pos(w1, w2, rv)

        # Plot satellites
        ax.scatter(pos[:, ep, 0], pos[:, ep, 1], pos[:, ep, 2], s=20, c='red', label='Satellites')
        
        # Plot motherships (last 3 positions)
        ax.scatter(pos[-3:, ep, 0], pos[-3:, ep, 1], pos[-3:, ep, 2], s=50, c='blue', marker='^', label='Motherships')
        
        # Plot rovers (last 4 positions before motherships)
        ax.scatter(pos[-7:-3, ep, 0], pos[-7:-3, ep, 1], pos[-7:-3, ep, 2], s=100, c='green', marker='s', label='Rovers')

        # Plot Earth
        r = self.Rp
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        ax.plot_surface(r * np.cos(u) * np.sin(v),
                        r * np.sin(u) * np.sin(v),
                        r * np.cos(v),
                        alpha=0.3, color="blue")

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'Constellation Configuration (Epoch {ep})')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([pos[:, ep, 0].max()-pos[:, ep, 0].min(),
                              pos[:, ep, 1].max()-pos[:, ep, 1].min(),
                              pos[:, ep, 2].max()-pos[:, ep, 2].min()]).max()
        mid_x = (pos[:, ep, 0].max()+pos[:, ep, 0].min()) * 0.5
        mid_y = (pos[:, ep, 1].max()+pos[:, ep, 1].min()) * 0.5
        mid_z = (pos[:, ep, 2].max()+pos[:, ep, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
        ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
        ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)

        return ax


    ###############################################################
    # STRING REPRESENTATION
    ###############################################################
    def __str__(self):
        return "QKD_Constellation_UDP_with_Penalty_Method" 