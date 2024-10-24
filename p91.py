import csv
import numpy as np
import random
from collections import deque
import heapq
import time
from datetime import datetime
import matplotlib.pyplot as plt


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_start_times(arrival_rate, num_events, start_time):
    start_hour, start_minute = map(int, start_time.split(':'))
    start_time_minutes = start_hour * 60 + start_minute
    inter_arrival_times = np.random.poisson(lam=arrival_rate, size=num_events)
    start_times = np.cumsum(inter_arrival_times) + start_time_minutes
    start_times %= (24 * 60)
    return start_times

def generate_service_times(service_rate, num_calls, max_duration):
    service_times = np.random.exponential(scale=1 / service_rate, size=num_calls)
    service_times = np.minimum(service_times, max_duration)
    return service_times

def get_free_channels(data, value):
    keys_less_than = []
    keys_greater_than = []
    for key, val in data.items():
        if val < value:
            keys_less_than.append(key)
        elif val > value:
            keys_greater_than.append(key)
    return keys_less_than, keys_greater_than

def get_next_channel(channels, channel_queue, base_state, r_value, p_value):
    for i, state in enumerate(base_state):
        new_channel = 0
        if state == 1:
            channel_queue[(i+1)] -= p_value
        elif state == 0:
            channel_queue[(i+1)] += r_value
            new_channel = i+1
            return new_channel
    return i+1

def find_samples(sample, ci):
    result = [sublist for sublist in sample if sublist[3] == ci]
    return result



device_coordinates = []
with open('device_coordinates.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        device_coordinates.append({
            'Device Number': row['Device Number'],
            'X Value': float(row['X Value']),
            'Y Value': float(row['Y Value'])
        })

device_pairs = [] # To store the distance between two devices

for i in range(len(device_coordinates)):
    for j in range(i + 1, len(device_coordinates)):
        device1 = device_coordinates[i]
        device2 = device_coordinates[j]
        distance = calculate_distance(device1['X Value'], device1['Y Value'], device2['X Value'], device2['Y Value'])
        device_pairs.append((device1['Device Number'], device2['Device Number'], round(distance, 2)))

'''
filename = "device_pairs.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Device 1', 'Device 2', 'Distance'])
    for row in device_pairs:
        writer.writerow(row)
file.close()
'''



devices = list(range(1, 101))
device_distance = {("D"+str(i)): [] for i in devices}

# Iterate through each pair
for device1, device2, distance in device_pairs:
    # Check if the distance is less than 30 vice versa
    if distance <= 30:
        device_distance[device1].append(device2)
        device_distance[device2].append(device1)

# Create a copy of the dictionary contaning empty list to iterate through
keys_to_remove = [key for key, value in device_distance.items() if len(value) == 0]

# Remove keys with empty lists from device_distance
for key in keys_to_remove:
    del device_distance[key]

print(device_distance)






# Parameter set
arrival_rate = 0.2
service_rate = 0.6
num_events = 100
start_time = '06:00'
max_duration = 10
total_devices = 100
max_attempts = 1000

start_times = generate_start_times(arrival_rate, num_events, start_time)
service_times = generate_service_times(service_rate, num_events, max_duration)
end_times = (start_times + service_times) % (24 * 60)

call_details = [] # To store the details of each call

for i, (start, duration) in enumerate(zip(start_times, service_times)):
    end = start + duration
    call_details.append((i + 1, start, end, duration))


call_data = [] # To convert timings of call_details

for call in call_details:
    call_id, start, end, duration = call

    start_hour, start_minute = divmod(start, 60)
    end_hour, end_minute = divmod(end, 60)
    duration_minutes, duration_seconds = divmod(int(duration * 60), 60)


    call_data.append((
        call_id,
        f"{int(start_hour):02d}:{int(start_minute):02d}",
        f"{int(end_hour):02d}:{int(end_minute):02d}",
        f"{duration_minutes:02d}:{duration_seconds:02d}"
))


'''
filename = "call_data.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Event', 'Start Time', 'End Time', 'Duration (min:sec)'])
    for row in call_data:
        writer.writerow(row)
file.close()
'''




call_channel = [] # To store the devices allocated for each call
count = 0

for call in call_details:
    _, start, end, duration = call
    selected_devices = []
    attempt_count = 0

    while len(selected_devices) < 2:
        device = random.randint(1, total_devices)
        if device not in selected_devices:
            selected_devices.append(device)
        attempt_count += 1
        if attempt_count >= max_attempts:
            break

    if len(selected_devices) < 2:
        print(f"Unable to find available devices for call {call[0]} within the maximum attempts.")
        continue

    device1, device2 = selected_devices
    device1 = "D"+str(device1)
    device2 = "D"+str(device2)
    distance = None

    for pair in device_pairs:
        if (device1, device2) == (pair[0], pair[1]) or (device1, device2) == (pair[1], pair[0]):
            distance = pair[2]
            count+=1
            break

    if distance is None:
        print(f"Distance not found for devices {device1} and {device2}.")
        continue

    call_channel.append(call + (device1, device2, distance))





call_distance = [] # To convert timings of call_channel

for call in call_channel:
    call_id, start, end, duration, device1, device2, distance = call

    start_hour, start_minute = divmod(start, 60)
    end_hour, end_minute = divmod(end, 60)
    duration_minutes, duration_seconds = divmod(int(duration * 60), 60)


    call_distance.append((
        call_id,
        f"{int(start_hour):02d}:{int(start_minute):02d}",
        f"{int(end_hour):02d}:{int(end_minute):02d}",
        f"{duration_minutes:02d}:{duration_seconds:02d}",
        f"{device1}",
        f"{device2}",
        distance
))



filename = "call_distance.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Event', 'Start Time', 'End Time', 'Duration (min:sec)', 'Device 1', 'Device 2', 'Distance'])
    for row in call_distance:
        writer.writerow(row)
file.close()







# Assuming channels are numbered from 1 to 14
channels = list(range(1, 15))
channel_end_times = [(0, i) for i in channels]
heapq.heapify(channel_end_times)
channel_free_times = {i: 0 for i in channels}

call_channel_distance = [] # To store the channel allocated for each call

close_calls = []

for call in call_channel:
    call_id, start, end, duration, device1, device2, distance = call
    if(float(call[-1])>30):
        _, channel = heapq.heappop(channel_end_times)
        start_time = max(channel_free_times[channel], start)
        end_time = start_time + duration
        heapq.heappush(channel_end_times, (end_time, channel))
        channel_free_times[channel] = end_time

        start_hour, start_minute = divmod(start_time, 60)
        end_hour, end_minute = divmod(end_time, 60)
        duration_minutes, duration_seconds = divmod(int(duration * 60), 60)

        call_channel_distance.append((
            call_id,
            #f"{int(start_hour):02d}:{int(start_minute):02d}",
            #f"{int(end_hour):02d}:{int(end_minute):02d}",
            #f"{duration_minutes:02d}:{duration_seconds:02d}",
            f"{start_time}",
            f"{end_time}",
            f"{duration}",
            f"{device1}",
            f"{device2}",
            distance,
            f"C{channel}"
        ))


    else:
        close_calls.append(call_id)
        end = start + duration

        start_hour, start_minute = divmod(start, 60)
        end_hour, end_minute = divmod(end, 60)
        duration_minutes, duration_seconds = divmod(int(duration * 60), 60)

        call_channel_distance.append((
            call_id,
            #f"{int(start_hour):02d}:{int(start_minute):02d}",
            #f"{int(end_hour):02d}:{int(end_minute):02d}",
            #f"{duration_minutes:02d}:{duration_seconds:02d}",
            f"{start_time}",
            f"{end_time}",
            f"{duration}",
            f"{device1}",
            f"{device2}",
            distance
        ))


#print(len(close_calls))
#print(close_calls)

#print(call_channel_distance)


filename = "call_channel_distance_1.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Event', 'Start Time', 'End Time', 'Duration (min:sec)', 'Device 1', 'Device 2', 'Distance', 'Channel'])
    for row in call_channel_distance:
        writer.writerow(row)
file.close()








# Assuming channels are numbered from 1 to 14
channels = list(range(1, 15))
channel_end_times = [(0, i) for i in channels]
heapq.heapify(channel_end_times)
channel_free_times = {i: 0 for i in channels}

device_channel_queue = []
for _ in range(100):
    device_channel_queue.append([0 for _ in range(14)])


call_channel_distance = [] # To store the channel allocated for each call

close_calls = []

free_channel_time = {i: 0 for i in channels}

channel_queue = {i: 0 for i in channels}
#print(channel_queue)

base_state = [0 for _ in range(14)]
#print(base_state)

device_state = [(0,) * 14 for _ in range(100)]
#print(device_state)

r_value = 0.8
p_value = 0.2


last_print_time = time.time()


for call in call_channel:
    call_id, start, end, duration, device1, device2, distance = call
    if float(call[-1]) > 30:
        _, channel = heapq.heappop(channel_end_times)
        start_time = max(channel_free_times[channel], start)
        end_time = start_time + duration
        heapq.heappush(channel_end_times, (end_time, channel))
        channel_free_times[channel] = end_time
        free_channel_time[channel] = float(end_time)
        free_channel_time = dict(sorted(free_channel_time.items(), key=lambda item: item[1]))

        ch_free, ch_occ = get_free_channels(free_channel_time, float(start_time))
        for i in ch_free:
            base_state[i - 1] = 0
        for i in ch_occ:
            base_state[i - 1] = 1
        for i in range(100):
            device_channel_queue[i][channel - 1] = 1

        start_hour, start_minute = divmod(start_time, 60)
        end_hour, end_minute = divmod(end_time, 60)
        duration_minutes, duration_seconds = divmod(int(duration * 60), 60)

        call_channel_distance.append((
            call_id,
            f"{start_time}",
            f"{end_time}",
            f"{duration}",
            f"{device1}",
            f"{device2}",
            distance,
            f"C{channel}"
        ))

    else:
        close_calls.append(call_id)
        channel_queue = dict(sorted(channel_queue.items(), key=lambda item: -item[1]))

        new_ch = 0
        next_channel = get_next_channel(channels, channel_queue, base_state, r_value, p_value)
        end = start + duration

        start_hour, start_minute = divmod(start, 60)
        end_hour, end_minute = divmod(end, 60)
        duration_minutes, duration_seconds = divmod(int(duration * 60), 60)

        call_channel_distance.append((
            call_id,
            f"{start_time}",
            f"{end_time}",
            f"{duration}",
            f"{device1}",
            f"{device2}",
            distance,
            f"C{next_channel}"
        ))

    # Check the time and print device_channel_queue every 10 seconds
    current_time = time.time()
    print(current_time)
    print(last_print_time)
    print(current_time - last_print_time)
    if current_time - last_print_time >= 10:
        print(device_channel_queue)
        last_print_time = current_time

print(device_channel_queue[0])
print(device_channel_queue[1])
print(device_channel_queue[2])
print(device_channel_queue[3])


print(len(close_calls))
print(close_calls)

#print(call_channel_distance)


filename = "call_channel_distance_2.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Event', 'Start Time', 'End Time', 'Duration (min:sec)', 'Device 1', 'Device 2', 'Distance', 'Channel'])
    for row in call_channel_distance:
        writer.writerow(row)
file.close()







call_drop_list = []
for i in call_channel_distance:
    call_drop_list.append([i[0], float(i[1]), float(i[2]), i[-1]])

#print(call_drop_list)

call_drop = []

for i in range(1, 15):
    channel = "C"+str(i)
    filtered_samples = find_samples(call_drop_list, channel)
    #print(filtered_samples)

    for j in range(len(close_calls)):
        for k in range(len(filtered_samples)):
            if(close_calls[j] == filtered_samples[k][0]):
                call_drop.append([filtered_samples[k], filtered_samples[k+1]])

#print(len(call_drop))
#print(call_drop)

list_of_drop = []
for i in range(len(call_drop)):
    if(call_drop[i][0][2] >= call_drop[i][1][1]):
        list_of_drop.append(call_drop[i][0][0])

print(len(list_of_drop))
print(sorted(list_of_drop))


# Graph plotting

channel_graph = []
base_call_graph = []
d2d_total_call_graph = []
d2d_call_graph = []
drop_call_graph = []


for i in range(1,15):
    ch_name = 'C' + str(i)
    #print(ch_name)

    base_call = 0
    d2d_total_call = 0
    d2d_call = 0
    drop_call = 0


    for i in range(num_events):
        if(call_channel_distance[i][-1] == ch_name and call_channel_distance[i][-2] > 30):
            base_call += 1
    
        elif(call_channel_distance[i][-1] == ch_name and call_channel_distance[i][-2] <= 30):
            if(call_channel_distance[i][0] not in list_of_drop):
                d2d_call += 1
            else:
                drop_call += 1
    

    d2d_total_call = d2d_call + drop_call

    channel_graph.append((base_call, d2d_total_call, d2d_call, drop_call))
    base_call_graph.append(base_call)
    d2d_total_call_graph.append(d2d_total_call)
    d2d_call_graph.append(d2d_call)
    drop_call_graph.append(drop_call)


#print(channel_graph)
print(base_call_graph)
print(d2d_total_call_graph)
print(d2d_call_graph)
print(drop_call_graph)

plt.figure(figsize=(10, 6))
plt.plot(base_call_graph, label='base_call', marker='o')
plt.plot(d2d_total_call_graph, label='d2d_total_call', marker='o')
plt.plot(d2d_call_graph, label='d2d_call', marker='o')
plt.plot(drop_call_graph, label='drop_call', marker='o')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('2D Line Graph')

# Adding a legend
plt.legend()

# Display the plot
plt.show()


# Number of bars
n = 14

# Create an array for the positions of the bars on the x-axis
index = np.arange(n)

# Define the width of the bars
bar_width = 0.2

# Plotting the data
fig, ax = plt.subplots(figsize=(12, 8))

# Create bars
bar1 = plt.bar(index, base_call_graph, bar_width, label='base_call')
bar2 = plt.bar(index + bar_width, d2d_total_call_graph, bar_width, label='d2d_total_call')
bar3 = plt.bar(index + 2 * bar_width, d2d_call_graph, bar_width, label='d2d_call')
bar4 = plt.bar(index + 3 * bar_width, drop_call_graph, bar_width, label='drop_call')

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Grouped Bar Graph')

# Adding a legend
plt.legend()

# Adding xticks
plt.xticks(index + bar_width * 1.5, range(1, n + 1))

# Display the plot
plt.show()
