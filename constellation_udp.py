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

