from packages import *

class Hubbard_mixed(CouplingModel, MPOModel):
    def __init__(self, model_param):
        
        ''' system size '''
        Lx = model_param["Lx"]
        Ly = model_param["Ly"]
#         Lx = q 
        
        ''' coupling constants'''
        t = model_param["t"]
        V = model_param["V"]
#         phi = model_param["phase"]
        
        
        ''' boundary conditions'''
        bc_MPS = model_param["bc_MPS"]
        bc_y = model_param["bc_y"]
        bc_x = model_param["bc_x"]
        
        
        ''' site with particle + U(1) symmetry conservation'''
        site = FermionSite(conserve='N', filling = 0.5)
        
        ''' define square lattice'''
        lat = Square(Lx, Ly, site, bc=[bc_x, bc_y], bc_MPS=bc_MPS)   
        

        CouplingModel.__init__(self, lat)
        
        dR1 = [1,0]
        
        if model_param["flux"]:
            
            '''hopping terms'''
            hops_along_x = (-t) * np.array([-1j + (-1) * np.exp(-1j * 2 * np.pi/Ly * np.arange(Ly)), -1j + np.exp(-1j * 2 * np.pi/Ly * np.arange(Ly))])
            hops_y = (-t) * np.array([2 * np.cos(2 * np.pi/Ly * np.arange(Ly)), - 2 * np.cos(2 * np.pi/Ly * np.arange(Ly))])
            
            self.add_coupling(hops_along_x, 0, 'Cd', 0,'C', dR1, plus_hc=True)
            self.add_onsite(hops_y, 0, 'N')
         
        else:
            
            '''hopping terms'''
            hops_along_x = (-t) * np.array([1 + (+1) * np.exp(-1j * 2 * np.pi/Ly * np.arange(Ly))])
#             hops_along_x = (-t)
            hops_y = (-t) * np.array([2 * np.cos(2 * np.pi/Ly * np.arange(Ly))])
            self.add_coupling(hops_along_x, 0, 'Cd', 0,'C', dR1, plus_hc=True)
            self.add_onsite(hops_y, 0, 'N')
        
        ''' density-density interaction'''
        for q1 in range(Ly):
            for q2 in range(Ly):
                dx1 = [0, q1]
                dx2 = [0, 0]
                dx4 = [0, q2]
                dx3 = [0, q2 - q1]

                self.add_multi_coupling(V/Ly, [('Cd', [0,q1], 0), ('C', [0,0], 0), ('Cd', [1,q2-q1], 0), ('C', [1,q2], 0)])
                self.add_multi_coupling(np.exp(-1j * 2 * np.pi/Ly * q1) * V/Ly, [('Cd', [0,q1], 0), ('C', [0,0], 0), ('Cd', [1,q2-q1], 0), ('C', [1,q2], 0)])

                if q1 == 0 and q2 == 0:
                    dd_interaction = V/Ly
                    self.add_onsite(dd_interaction, 0, 'N')

                else:
                    self.add_multi_coupling(np.cos(2 * np.pi/Ly * q1) * V/Ly, [('Cd', dx1, 0), ('C', dx2, 0), ('Cd', dx3, 0), ('C', dx4, 0)])                    
        
        MPOModel.__init__(self, lat, self.calc_H_MPO())

# if flux = True, Lx = 2
# if flux = False, Lx = 1

model_param = {"Lattice" : Square,
              "Ly" : 3,
              "Lx" : 3,
              "t" : 1.0,
              "V" : float(sys.argv[1]),
              "fraction" : 2,
              "bc_MPS" : "infinite",
              "bc_y" : 'periodic',
              "bc_x" : 'periodic',
              "flux" : False}

# chi_list = tenpy.algorithms.dmrg.chi_list(500, dchi=200, nsweeps=20)
chi_list = {0:200, 20:400, 40:800}
dmrg_params = {"trunc_params": {"chi_max": 1000, "svd_min": 1.e-10}, "mixer": True, "chi_list" : chi_list,"max_sweeps": 100}

M = Hubbard_mixed(model_param)
sites = M.lat.mps_sites()
psi = MPS.from_product_state(sites,["full","empty","empty","full","empty","empty", "full","empty","empty"],"infinite")

info = dmrg.run(psi, M, dmrg_params)
energy = info['E']
delta_energy = abs(info["sweep_statistics"]['E'][-1] - info["sweep_statistics"]['E'][-2])

data = {'psi': psi, 'dmrg_params': dmrg_params, 'Energy': energy, 'Delta Energy': delta_energy, 'model_params': model_param}

filename = 'spinless_without_conserve_ky_V_' + str(model_param["V"]) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump(data, f)