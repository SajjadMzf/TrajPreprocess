import numpy as np
import coordinate_functions as cf
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.backends.backend_pdf import PdfPages

font = {'size'   : 24}
matplotlib.rcParams['figure.figsize'] = (16, 13)
matplotlib.rc('font', **font)
x = np.arange(0,1000,1)*10*np.pi
x = x/1000
x_shifted = x 
y = np.sin(x)
y_shifted = np.sin(x_shifted)+0.5

traj = np.stack((x_shifted, y_shifted), axis =1)
traj = traj[400:800]
ref = np.stack((x, y), axis =1)

traj_f, ref_f = cf.cart2frenet(traj, ref)
traj_fc = cf.frenet2cart(traj_f, ref)

with PdfPages('frenet_cart.pdf') as export_pdf:
    plt.figure(1)

    plt.subplot(211)
    plt.ylabel('Y(m)')
    plt.xlabel('X(m)')
    plt.title('Trajectory and Reference Line in Cartesian Coordinate System')
    plt.plot(traj[:,0], traj[:,1], color = 'b', marker ='.', label='Trajectory (Cartesian)')
    plt.plot(ref[:,0], ref[:,1], color = 'r',marker ='.',label='Reference (Cartesian)')
    plt.ylim(-2, 2)
    #plt.plot(traj_fc[:,0], traj_fc[:,1], color = 'g',marker ='.')
    plt.legend(loc='lower right')
    plt.grid()
    plt.subplot(212)
    plt.ylabel('Cross Track(m)')
    plt.xlabel('Along Track(m)')
    plt.title('Trajectory and Reference Line in Frenet Coordinate System')
    plt.plot(traj_f[:,0], traj_f[:,1], color = 'g',marker ='.', label='Trajectory (Frenet)')
    plt.plot(ref_f[:,0], ref_f[:,1], color = 'r',marker ='.', label='Reference (Frenet)')
    plt.ylim(-2, 2)
    plt.legend(loc='lower right')
    
    plt.grid()
    plt.tight_layout()
    export_pdf.savefig()

