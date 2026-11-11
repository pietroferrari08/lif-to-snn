import matplotlib.pyplot as plt

#modello comportamento del neurone

#COSTANTI DEL NEURONE 
BETA = 0.8
J_values = [0.15, 0.25, 0.4, 0.6]
THRESHOLD = 1
STEP = 100
time_steps = range(1, STEP + 1)

for current_J in J_values:
    print("\nSimulazione con J =", current_J)

    #variabili del modello
    potentials = []     
    spikes = []
    spike_times = []
    spike_potentials = []
    potentials_after_reset = []
    isi = []
    U = 0

    for i in range(1,STEP + 1):
        U = U*BETA + current_J
        potentials.append(U)
        #DEBUG print("il valore di U è: ", U)
        if U >= THRESHOLD:
            if spike_times: #verifica se la lista contiene almeno 1 elemento
                isi.append(i - spike_times[-1])
            spike_potentials.append(U)
            U = 0
            spikes.append(1)
            spike_times.append(i)
            #DEBUG: print('SPIKE')
        else :
            spikes.append(0)
        potentials_after_reset.append(U)
    num_spikes = sum(spikes)
    firing_rate = num_spikes/STEP #spike per time step

    #DEBUG: print(isi)
    #DEBUG: print(potentials)
    #DEBUG: print(spikes)

    print("numero di spikes: " + str(num_spikes))
    print("firing rate medio: " + str(firing_rate))

    if len(isi) > 0:
        mean_isi = sum(isi) / len(isi)
        print("ISI medio:", mean_isi)
    else:
        print("ISI medio non definito")
    #treno di impulsi
    plt.figure(1)
    plt.xlabel('Time step')
    plt.ylabel('Input J')
    plt.title('LIF neuron - Spike trains for different inputs')
    plt.yticks(J_values)
    plt.xlim(0, STEP + 1)
    plt.grid(axis='x', alpha=0.2)
    plt.scatter(spike_times, [current_J] * len(spike_times))
    
##visualizzazione grafici


#andamento approssimato del potenziale di membrana

plt.figure(2)
plt.plot(time_steps, potentials, label='Membrane potential')
plt.axhline(y=THRESHOLD, color='red', linestyle='--',
            label='Threshold')
plt.scatter(spike_times, spike_potentials, color='orange',label='Spike')
plt.xlabel('Time step')
plt.ylabel('Membrane potential')
plt.title('Leaky Integrate-and-Fire neuron')
plt.legend()

#andamento effettivo del potenziale di membrana

plt.figure(3)
plt.plot(time_steps, potentials, marker='o', linestyle='None')
plt.plot(time_steps, potentials_after_reset, marker='x', linestyle='None')
plt.vlines(spike_times, 0, spike_potentials, color='orange')
plt.title('LIF - Reset visualization')

plt.figure(4)
plt.stem(time_steps,spikes)

plt.show()
