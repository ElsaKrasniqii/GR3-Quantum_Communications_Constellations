import pykep as pk
import numpy as np
import scipy
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sgp4.api import Satrec, SatrecArray
from sgp4.api import WGS72
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


###############################################################
#  MOTHERSHIP TLES
###############################################################
def get_mothership_satellites():
    mothership_tles = [
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
        ],
        [
            "1 40128U 14050A   22349.31276420 -.00000077  00000-0  00000-0 0  9995",
            "2 40128  50.1564 325.0733 1614819 130.5958 244.6527  1.85519534 54574"
        ],
        [
            "1 49810U 21116B   23065.71091236 -.00000083  00000+0  00000+0 0  9998",
            "2 49810  57.2480  13.9949 0001242 301.4399 239.8890  1.70475839  7777"
        ],
        [
            "1 44878U 19092F   22349.75758852  .00015493  00000-0  00000-0 0  9998",
            "2 44878  97.4767 172.6133 0012815  68.6990 291.5614 15.23910904165768"
        ],
        [
            "1 04382U 70034A   22349.88472104  .00001138  00000-0  18306-3 0  9999",
            "2 04382  68.4200 140.9159 1043234  48.2283 320.3286 13.08911192477908"
        ]
    ]
    return [Satrec.twoline2rv(t[0], t[1]) for t in mothership_tles]


###############################################################
#  CLASS: CONSTELLATION UDP
###############################################################
class constellation_udp:

    def __init__(self):
        self._t0 = 10000
        self.n_epochs = 11
        self._duration = 10

        jd0, fr = pk.epoch(self._t0, 'mjd2000').jd, 0.0
        self.jds = np.linspace(jd0, jd0 + self._duration * 365.25, self.n_epochs)
        self.frs = np.zeros_like(self.jds)  # Fixed: all fractional parts to 0
        self.ep_ref = pk.epoch_from_iso_string("19491231T000000")

        # MOTHERSHIPS
        motherships = get_mothership_satellites()
        self.pos_m = self.construct_mothership_pos(SatrecArray(motherships))
        self.n_motherships = len(motherships)

        # ROVERS
        rover_path = "./data/spoc2/constellations/rovers.txt"
        if not os.path.exists(rover_path):
            raise FileNotFoundError(f"Rover file not found: {rover_path}")
        
        self.rovers_db = np.loadtxt(rover_path)
        if len(self.rovers_db.shape) < 2 or self.rovers_db.shape[1] < 2:
            raise ValueError("Rovers file must have at least 2 columns")
        
        self.lambdas = self.rovers_db[:, 0]
        self.phis = self.rovers_db[:, 1]
        self.n_rovers = 4
        self._min_rover_dist = 3000

        # CONSTANTS
        self.LOS = 1.05 * pk.EARTH_RADIUS / 1000
        self.R_p = pk.EARTH_RADIUS / 1000
        self.w_p = 7.29e-5
        self.eps_z = np.cos(np.pi / 3)
        self._min_sat_dist = 50


    ###############################################################
    #   POSITION FUNCTIONS
    ###############################################################
    def construct_mothership_pos(self, motherships):
        err, pos, _ = motherships.sgp4(self.jds, self.frs)
        if not np.all(err == 0):
            raise ValueError("Mothership SGP4 propagation failed")
        return pos


    def construct_rover_pos(self, lambda0, phi0):
        pos = np.zeros((self.n_rovers, self.n_epochs, 3))
        times = (self.jds - self.jds[0]) * 24 * 3600

        for i, t in enumerate(times):
            # Fixed: Corrected spherical to Cartesian conversion
            pos[:, i, 0] = self.R_p * np.cos(lambda0) * np.cos(phi0 + self.w_p * t)
            pos[:, i, 1] = self.R_p * np.cos(lambda0) * np.sin(phi0 + self.w_p * t)
            pos[:, i, 2] = self.R_p * np.sin(lambda0)

        return pos


    ###############################################################
    #   GENERATE WALKER CONSTELLATIONS
    ###############################################################
    def generate_walker(self, S, P, F, a, e, incl, w, t0):
        sats = []
        # a is in km, convert to Earth radii for SGP4
        a_earth_radii = a / (pk.EARTH_RADIUS / 1000)
        mm = np.sqrt(pk.MU_EARTH / (a_earth_radii * pk.EARTH_RADIUS)**3) * 60  # rad/min

        for i in range(P):
            for j in range(S):
                sat = Satrec()
                # Convert angles to degrees for SGP4
                incl_deg = np.degrees(incl)
                w_deg = np.degrees(w)
                raan = np.degrees(2 * np.pi / P * i)
                m = np.degrees(2*np.pi/P/S * F * i + 2*np.pi/S * j)
                
                sat.sgp4init(
                    WGS72,           # gravity model
                    'i',             # 'a' = afspc mode, 'i' = improved mode
                    j + i*S,         # satnum: Satellite number
                    t0,              # epoch: days since 1949-12-31 00:00 UT
                    0.0,             # bstar: drag coefficient
                    0.0,             # ndot: ballistic coefficient
                    0.0,             # nddot: second derivative of mean motion
                    e,               # eccentricity
                    w_deg,           # argpo: argument of perigee (degrees)
                    incl_deg,        # inclo: inclination (degrees)
                    m,               # mo: mean anomaly (degrees)
                    mm,              # no_kozai: mean motion (rad/min)
                    raan             # nodeo: RA of ascending node (degrees)
                )
                sats.append(sat)

        return SatrecArray(sats)


    ###############################################################
    #   COMBINE POSITIONS
    ###############################################################
    def construct_pos(self, w1, w2, rovers):
        err1, p1, _ = w1.sgp4(self.jds, self.frs)
        err2, p2, _ = w2.sgp4(self.jds, self.frs)

        if not np.all(err1 == 0) or not np.all(err2 == 0):
            print(f"Walker errors: w1={np.sum(err1 != 0)}, w2={np.sum(err2 != 0)}")
            # Don't raise, just continue with valid positions

        # Combine all positions: w1, w2, motherships, rovers
        all_pos = []
        if len(p1) > 0:
            all_pos.append(p1)
        if len(p2) > 0:
            all_pos.append(p2)
        all_pos.append(self.pos_m)
        all_pos.append(rovers)
        
        return np.concatenate(all_pos, axis=0)


    ###############################################################
    #   CONSTRAINT FUNCTIONS
    ###############################################################
    def get_rover_constraint(self, lambda0, phi0):
        # Calculate current positions at time 0
        pos = np.zeros((self.n_rovers, 3))
        t = 0
        pos[:, 0] = self.R_p * np.cos(lambda0) * np.cos(phi0 + self.w_p * t)
        pos[:, 1] = self.R_p * np.cos(lambda0) * np.sin(phi0 + self.w_p * t)
        pos[:, 2] = self.R_p * np.sin(lambda0)

        # Calculate angular distance
        def angular_distance(u, v):
            # u and v are unit vectors from center
            u_norm = u / np.linalg.norm(u)
            v_norm = v / np.linalg.norm(v)
            dot = np.clip(np.dot(u_norm, v_norm), -1.0, 1.0)
            return np.arccos(dot)

        d = scipy.spatial.distance.cdist(
            pos, pos,
            lambda u, v: pk.EARTH_RADIUS/1000 * angular_distance(u, v)
        )
        np.fill_diagonal(d, np.inf)
        min_dist = np.min(d)
        return self._min_rover_dist - min_dist, min_dist


    def get_sat_constraint(self, dmin):
        if dmin == np.inf:
            return 0  # No valid distances found
        return self._min_sat_dist - dmin


    ###############################################################
    #   GRAPH FUNCTIONS
    ###############################################################
    def line_of_sight(self, r1, r2):
        d = np.linalg.norm(r2 - r1)
        if d < 1e-6:
            return np.linalg.norm(r1)
        r21 = (r2 - r1) / d
        h1 = np.dot(r1, r21)
        arg = np.linalg.norm(r1)**2 - h1**2
        return np.sqrt(max(arg, 0))


    def zenith_angle(self, src, dst):
        d = dst - src
        d_norm = np.linalg.norm(d)
        src_norm = np.linalg.norm(src)
        if d_norm < 1e-6 or src_norm < 1e-6:
            return 0
        return np.dot(d, src) / (d_norm * src_norm)


    def qkd_metric(self, idx, src, dst, cosz, eta):
        ew = -np.log(max(eta, 1e-10))
        d = np.linalg.norm(src - dst)
        ew += 2 * np.log(max(d, 1e-3))
        ew = max(ew, 1e-3)

        # Rover-specific penalty
        if idx <= self.n_rovers:
            if cosz >= self.eps_z:
                zenith = np.pi/2 - np.arccos(min(max(cosz, -1), 1))
                ew += 1.0 / max(np.sin(zenith), 1e-3)
            else:
                ew = 0
        return ew, d


    ###############################################################
    def build_graph(self, ep, pos, n1, eta):
        N = pos.shape[0]
        adj = np.zeros((N, N))
        dmin = np.inf

        for i in range(N):
            for j in range(i):
                los = self.line_of_sight(pos[i], pos[j])
                cosz = self.zenith_angle(pos[i], pos[j])

                if los < self.LOS and cosz <= 0:  # Fixed condition
                    et = eta[0] if j < n1 else eta[1]
                    adj[i, j], d = self.qkd_metric(N-i, pos[i], pos[j], cosz, et)
                    dmin = min(dmin, d)
                    adj[j, i] = adj[i, j]

        return nx.from_numpy_array(adj), adj, dmin


    ###############################################################
    def average_shortest_path(self, G, nm, nr, ep, verbose=False):
        r = 0
        N = len(G.nodes())

        for i in range(nr):
            for j in range(nm):
                try:
                    path_len = nx.shortest_path_length(
                        G,
                        N - nm - nr + j,
                        N - nr + i,
                        weight="weight",
                        method="dijkstra"
                    )
                    r += path_len
                except nx.NetworkXNoPath:
                    if verbose:
                        print(f"No path between mothership {j} and rover {i}")
                    r += 1e4

        return r / (nm * nr) if nm * nr > 0 else 1e4


    ###############################################################
    #   CONSTRUCT WALKERS
    ###############################################################
    def construct_walkers(self, x):
        # Extract parameters
        a1, e1, i1, w1, eta1, a2, e2, i2, w2, eta2, S1, P1, F1, S2, P2, F2, *rovers = x
        
        # Convert to integers
        S1, P1, F1 = int(S1), int(P1), int(F1)
        S2, P2, F2 = int(S2), int(P2), int(F2)
        
        # Generate walker constellations
        w1_arr = self.generate_walker(S1, P1, F1, a1, e1, i1, w1, self._t0)
        w2_arr = self.generate_walker(S2, P2, F2, a2, e2, i2, w2, self._t0)
        
        return w1_arr, w2_arr


    ###############################################################
    #   FITNESS FUNCTION
    ###############################################################
    def fitness(self, x, verbose=False):
        w1, w2 = self.construct_walkers(x)
        
        # Extract parameters
        a1, e1, i1, w1_angle, eta1, a2, e2, i2, w2_angle, eta2, S1, P1, F1, S2, P2, F2, *rovers = x
        S1, P1, S2, P2 = int(S1), int(P1), int(S2), int(P2)
        
        N1 = S1 * P1
        N2 = S2 * P2

        # Get rover indices (ensure they're valid)
        rover_indices = [int(r) % len(self.lambdas) for r in rovers[:4]]
        lamb0 = self.lambdas[rover_indices]
        phi0 = self.phis[rover_indices]
        
        rovers_pos = self.construct_rover_pos(lamb0, phi0)
        pos = self.construct_pos(w1, w2, rovers_pos)

        f1 = 0
        dmin_ep = np.inf
        valid_epochs = 0

        for ep in range(1, self.n_epochs):
            G, _, dmin = self.build_graph(ep, pos[:, ep, :], N1, (eta1, eta2))
            dmin_ep = min(dmin_ep, dmin)
            
            avg_path = self.average_shortest_path(G, self.n_motherships, self.n_rovers, ep, verbose)
            f1 += avg_path
            valid_epochs += 1

        f1 = f1 / max(valid_epochs, 1)
        f2 = eta1 * N1 + eta2 * N2

        rc, rdm = self.get_rover_constraint(lamb0, phi0)
        sc = self.get_sat_constraint(dmin_ep)

        if verbose:
            print(f"Fitness components: f1={f1:.2f}, f2={f2:.2f}, rc={rc:.2f}, sc={sc:.2f}")
            print(f"Rover min distance: {rdm:.2f} km, Satellite min distance: {dmin_ep:.2f} km")

        return [f1/34, f2/100000, rc, sc]


    ###############################################################
    #   EXAMPLE INPUT
    ###############################################################
    def example(self, verbose=False):
        # Parameters: a,e,i,w,eta for walker1, same for walker2, 
        # S,P,F for walker1, S,P,F for walker2, 4 rover indices
        return [7000, 0.001, 1.2, 0, 55, 
                8000, 0.001, 1.2, 0, 15, 
                10, 2, 1, 
                10, 2, 1, 
                13, 21, 34, 55]


    ###############################################################
    #   PLOTTING
    ###############################################################
    def plot(self, x, src=1, dst=1, ep=1, lims=10000):
        sns.set(style="darkgrid")
        plt.style.use("dark_background")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        w1, w2 = self.construct_walkers(x)

        # Get rover positions
        rover_indices = [int(r) % len(self.lambdas) for r in x[-4:]]
        lamb0 = self.lambdas[rover_indices]
        phi0 = self.phis[rover_indices]
        rv = self.construct_rover_pos(lamb0, phi0)
        
        # Get all positions
        pos = self.construct_pos(w1, w2, rv)

        # Plot satellites (different colors for different groups)
        total_sats = pos.shape[0]
        colors = ['blue'] * (int(x[10]) * int(x[11])) + \
                 ['green'] * (int(x[14]) * int(x[15])) + \
                 ['red'] * self.n_motherships + \
                 ['yellow'] * self.n_rovers
        
        ax.scatter(pos[:, ep, 0], pos[:, ep, 1], pos[:, ep, 2], 
                  s=20, c=colors[:total_sats], alpha=0.7)

        # Earth
        r = pk.EARTH_RADIUS/1000
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
        x_earth = r * np.cos(u) * np.sin(v)
        y_earth = r * np.sin(u) * np.sin(v)
        z_earth = r * np.cos(v)
        ax.plot_surface(x_earth, y_earth, z_earth,
                        alpha=0.2, color="blue", edgecolor='none')

        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
        ax.set_zlim(-lims, lims)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'Constellation at epoch {ep}')
        
        return ax


###############################################################
#   MAIN TEST RUN
###############################################################
if __name__ == "__main__":
    print("\n=== RUNNING CONSTELLATION UDP ===\n")

    try:
        udp = constellation_udp()
        x = udp.example(verbose=True)
        
        print(f"Example parameters: {x}")
        print(f"Number of rovers in database: {len(udp.lambdas)}")

        print("\nComputing fitness...")
        fitness = udp.fitness(x, verbose=True)
        print(f"Fitness values: {fitness}")

        print("\nPlotting...")
        ax = udp.plot(x, src=1, dst=1, ep=1, lims=20000)
        plt.show()

        print("\n=== FINISHED SUCCESSFULLY ===")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the rovers.txt file exists in ./data/spoc2/constellations/")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

        