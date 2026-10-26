import matplotlib.pyplot as plt

#modello comportamento del neurone

#COSTANTI DEL NEURONE 
BETA = 0.8
J = 0.4
THRESHOLD = 1

#ciclo for per aggiornare
potentials = []     
spikes = []
spike_times = []
spike_potentials = []
potentials_after_reset = []
U = 0

for i in range(1,16):
    U = U*BETA + J
    potentials.append(U)
    print("il valore di U è: ", U)
    if U >= THRESHOLD:
        spike_potentials.append(U)
        U = 0
        spikes.append(1)
        spike_times.append(i)
        print('SPIKE')
    else :
        spikes.append(0)
    potentials_after_reset.append(U)
    
print(potentials)
print(spikes)

time_steps = range(1, len(potentials) + 1)


plt.figure(1)
plt.plot(time_steps, potentials, label='Membrane potential')
plt.axhline(y=THRESHOLD, color='red', linestyle='--',
            label='Threshold')
plt.scatter(spike_times, spike_potentials, color='orange',label='Spike')
plt.xlabel('Time step')
plt.ylabel('Membrane potential')
plt.title('Leaky Integrate-and-Fire neuron')
plt.legend()


plt.figure(2)
plt.plot(time_steps, potentials, marker='o', linestyle='None')
plt.plot(time_steps, potentials_after_reset, marker='x', linestyle='None')
plt.vlines(spike_times, 0, spike_potentials, color='orange')
plt.title('LIF - Reset visualization')


plt.show()