from packages import *

class Spinless_real(CouplingModel, MPOModel):
    def __init__(self, model_param):
        
        ''' system size '''
        Lx = model_param["Lx"]
        Ly = model_param["Ly"]
        
        ''' coupling constants'''
        t = model_param["t"]
        V = model_param["V"]
                
        ''' boundary conditions'''
        bc_MPS = model_param["bc_MPS"]
        bc_y = model_param["bc_y"]
        bc_x = model_param["bc_x"]
        
        ''' site with particle + U(1) symmetry conservation'''
        site = FermionSite(conserve='N', filling = 0.5)
        
        ''' define triangular lattice'''
        lat = Triangular(Lx, Ly, site, bc=[bc_x, bc_y], bc_MPS=bc_MPS)   
        
        CouplingModel.__init__(self, lat)
        
        dR1 = [1,0]
        dR2 = [0,1]
        dR2m1 = [1,-1]
        
        if model_param["flux"]:
            
            '''hoppings along x'''
            hops_along_x = (-t) * np.array([[-1j] * Ly, [-1j] * Ly])
            self.add_coupling(hops_along_x, 0, 'Cd', 0,'C', dR1, plus_hc=True)
            
            ''' density-density interaction along x'''
            self.add_coupling(V, 0, 'N', 0,'N', dR1)
            
            
            
            '''hoppings along y'''
            hops_along_y = (-t) * np.array([[1] * Ly, [-1] * Ly])
            self.add_coupling(hops_along_y,0,'Cd',0,'C',dR2, plus_hc=True)
            
            ''' density-density interaction along y'''
            self.add_coupling(V, 0, 'N', 0,'N', dR2)
            
            
            '''hoppings along diag'''
            hops_diag = (-t) * np.array([[-1] * Ly, [1] * Ly])
            self.add_coupling(hops_diag,0,'Cd',0,'C', dR2m1, plus_hc=True)
            
            ''' density-density interaction along diag'''
            self.add_coupling(V, 0, 'N', 0,'N', dR2m1)
        
        else:
            
            '''hoppings along x'''
            hops_along_x = (-t) 
            self.add_coupling(hops_along_x, 0, 'Cd', 0,'C', dR1, plus_hc=True)
            
            ''' density-density interaction along x'''
            self.add_coupling(V, 0, 'N', 0,'N', dR1)
            
            
            
            '''hoppings along y'''
            hops_along_y = (-t) 
            self.add_coupling(hops_along_y,0,'Cd',0,'C',dR2, plus_hc=True)
            
            ''' density-density interaction along y'''
            self.add_coupling(V, 0, 'N', 0,'N', dR2)
            
            
            '''hoppings along diag'''
            hops_diag = (-t) 
            self.add_coupling(hops_diag,0,'Cd',0,'C', dR2m1, plus_hc=True)
            
            ''' density-density interaction along diag'''
            self.add_coupling(V, 0, 'N', 0,'N', dR2m1)
            

        MPOModel.__init__(self, lat, self.calc_H_MPO())

        
        
# if flux = True, put Lx = 2
# if flux = False, put Lx = 1

model_param = {"Lattice" : Triangular,
              "Ly" : 3,
              "Lx" : 3,
              "t" : 1.0,
              "V" : float(sys.argv[1]),
              "fraction" : 2,
              "bc_MPS" : "infinite",
              "bc_y" : 'periodic',
              "bc_x" : 'periodic',
              "flux" : False}

# chi_list = tenpy.algorithms.dmrg.chi_list(1000, dchi=500, nsweeps=20)
chi_list = {0:200, 20:400, 40:800}
dmrg_params = {"trunc_params": {"chi_max": 1000, "svd_min": 1.e-10}, "mixer": True, "chi_list" : chi_list,"max_sweeps": 90}
 

M = Spinless_real(model_param)
sites = M.lat.mps_sites()
psi = MPS.from_product_state(sites,["full","full","full",
                                    "empty","empty","empty",
                                    "empty","empty","empty"],"infinite")

info = dmrg.run(psi, M, dmrg_params)

energy = info['E']
delta_energy = abs(info["sweep_statistics"]['E'][-1] - info["sweep_statistics"]['E'][-2])
data = {'psi': psi, 'dmrg_params': dmrg_params, 'Energy': energy, 'Delta Energy': delta_energy, 'model_params': model_param}

filename = '6_6_data_spinless_real_V_' + str(model_param["V"]) + '.pkl'
with open(filename, 'wb') as f:
    pickle.dump(data, f)