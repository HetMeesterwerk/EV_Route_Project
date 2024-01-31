import numpy as np
import random
import time
import psutil
from time import time as current_time
from deap import base, creator, tools
import googlemaps
import logging
import math
from truck import Truck
from global_cache import global_distance_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
gmaps = googlemaps.Client(key='Google Key Blocked')

class PSO:
    def _init_(self, num_trucks, warehouse, locations, charging_stations, temperature, n_particles=10, iters=5, consider_traffic=True):
        # Set up the PSO parameters and initialize the DEAP genetic algorithm library
        self.num_trucks = num_trucks
        self.warehouse = warehouse
        self.locations = locations
        self.charging_stations = charging_stations
        self.n_particles = n_particles
        self.iters = iters
        self.consider_traffic = consider_traffic
        self.temperature = temperature
        self.setup_deap()
        logging.info("Initialized PSO with {} trucks, {} particles, and {} iterations".format(num_trucks, n_particles, iters))

    def setup_deap(self):
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Particle"):
            creator.create("Particle", list, fitness=creator.FitnessMin, speed=None, 
                        pbest=None, best=None)

        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generate_particle)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.update_particle)
        self.toolbox.register("evaluate", self.route_fitness)

    def generate_particle(self):
        # Randomly assign each location to a truck
        particle = [random.randint(0, self.num_trucks - 1) for _ in range(len(self.locations))]

        # Shuffle to add randomness
        random.shuffle(particle)

        # Ensure all trucks are represented
        represented_trucks = set(particle)
        missing_trucks = set(range(self.num_trucks)) - represented_trucks

        # Assign missing trucks to random locations
        for truck in missing_trucks:
            random_location = random.randint(0, len(self.locations) - 1)
            particle[random_location] = truck

        # Create particle object
        particle = creator.Particle(particle)
        particle.speed = [random.uniform(-1, 1) for _ in particle]
        particle.pbest = None
        particle.best = None
        return particle
    
        # Convert a dictionary location to a string or a tuple for caching
    def location_to_key(self, location):
        if isinstance(location, dict):
            # Assuming the dictionary has 'lat' and 'lng' as keys
            return (location['lat'], location['lng'])
        return location  # If it's already a suitable type

    def update_particle(self, particle, best, phi1=2.05, phi2=2.05):
        if particle.pbest is None or best is None:
            return  # Skip update if pbest or best is not set

        u1 = [phi1 * random.uniform(0, 1) for _ in range(len(particle))]
        u2 = [phi2 * random.uniform(0, 1) for _ in range(len(particle))]
        v = [particle.speed[i] + u1[i] * (particle.pbest[i] - particle[i]) + u2[i] * (best[i] - particle[i]) for i in range(len(particle))]
        particle.speed = v
        particle[:] = [particle[i] + particle.speed[i] for i in range(len(particle))]

    def calculate_route_cost(self, route, charging_stations, temperature):
        """
        Calculate the cost of a given route considering the need to charge the truck.
        """
        total_cost = 0
        total_time = 0
        visited_charging_stations = []
        truck = Truck(400, 0.8)  # Adjust as per your Truck model

        for i in range(len(route) - 1):
            start, end = route[i], route[i + 1]

            # Check if the truck can proceed to the next location and then to a charging station if needed
            can_proceed, nearest_station, distance_to_station = self.can_proceed_to_next_with_charging_option(truck, start, end, charging_stations, temperature)

            if can_proceed:
                distance_to_next_location, time = self.calculate_distance_and_time(start, end)
                total_cost += distance_to_next_location
                total_time += time
                truck.update_battery_level(distance_to_next_location, temperature)
                if nearest_station:
                    total_cost += distance_to_station
                    truck.charge_battery()
                    visited_charging_stations.append(nearest_station)
            else:
                # Divert to nearest charging station from current location before proceeding to next location
                nearest_station_from_current, distance_to_station_from_current = self.find_nearest_charging_station(start, charging_stations)
                if nearest_station_from_current is not None:
                    total_cost += distance_to_station_from_current
                    truck.charge_battery()
                    visited_charging_stations.append(nearest_station_from_current)

                    # Now proceed to the next location
                    distance_to_next_location, time = self.calculate_distance_and_time(nearest_station_from_current, end)
                    total_cost += distance_to_next_location
                    total_time += time
                    truck.update_battery_level(distance_to_next_location, temperature)
                else:
                    logging.error(f"Cannot proceed from {start} to {end}. Insufficient battery and no charging station nearby.")
                    return float('inf'), visited_charging_stations

            # Add traffic delay cost if considered
            if self.consider_traffic:
                total_cost += self.get_traffic_delay(start, end)

        logging.info(f"Total route cost: {total_cost} km, visited charging stations: {visited_charging_stations}")
        return total_cost, total_time, visited_charging_stations
    
    def can_proceed_to_next_with_charging_option(self, truck, current_location, next_location, charging_stations, temperature):
        """
        Check if the truck can proceed to the next location and then reach a charging station if needed.
        """
        # Calculate distance from the current location to the next location
        distance_to_next_location, _ = self.calculate_distance_and_time(current_location, next_location)

        # Check if the truck can reach the next location with the minimum required battery
        if not truck.can_reach_with_min_battery(distance_to_next_location, temperature):
            return False, None, 0

        # Project the truck's battery level after reaching the next location
        projected_battery_level = truck.project_battery_level_after_distance(distance_to_next_location, temperature)

        # Find the nearest charging station from the next location
        nearest_station, distance_to_station = self.find_nearest_charging_station(next_location, charging_stations)

        # Check if the truck can reach the nearest charging station from the next location
        if projected_battery_level >= truck.calculate_required_battery_level(distance_to_station):
            return True, nearest_station, distance_to_station
        else:
            return False, None, 0

    def euclidean_distance(self, lat1, lng1, lat2, lng2):
        # Calculate Euclidean distance using latitude and longitude
        R = 6371  # Radius of the Earth in km
        dLat = math.radians(lat2 - lat1)
        dLng = math.radians(lng2 - lng1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLng/2) * math.sin(dLng/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance
        
    def find_nearest_charging_station(self, current_location, charging_stations):
        try:
            current_key = self.location_to_key(current_location)
            current_lat, current_lng = global_distance_cache.get(current_key, (None, None))
            if current_lat is None or current_lng is None:
                geocode_result = gmaps.geocode(current_location)
                if not geocode_result:
                    raise ValueError(f"Geocoding failed for location: {current_location}")

                current_lat = geocode_result[0]['geometry']['location']['lat']
                current_lng = geocode_result[0]['geometry']['location']['lng']
                global_distance_cache[current_key] = (current_lat, current_lng)

            nearest_station, min_distance = None, float('inf')
            for station in charging_stations:
                station_key = self.location_to_key(station)
                distance = self.euclidean_distance(current_lat, current_lng, station_key[0], station_key[1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_station = station

            if nearest_station is None:
                raise ValueError("No nearby charging station found.")

            return nearest_station, min_distance
        except Exception as e:
            logging.error(f"Error in find_nearest_charging_station: {e}")
            return None, float('inf')

    def optimize_route(self):
        logging.info("Starting route optimization...")
        start_time = time.time()
        process = psutil.Process()
        population = self.toolbox.population(n=self.n_particles)
        best = None

        for it in range(self.iters):
            logging.info(f"Generation {it}")
            for part in population:
                fitness, visited_stations = self.toolbox.evaluate(part)
                part.fitness.values = (fitness,)  # Update fitness value
                if not part.best or part.best.fitness.values[0] > fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = (fitness,)
                if best is None or best.fitness.values[0] > fitness:
                    best = creator.Particle(part)
                    best.fitness.values = (fitness,)
                    best_visited_stations = visited_stations  # Capture the visited stations for the best particle

            for part in population:
                self.toolbox.update(part, best)

        end_time = time.time()
        computation_time = end_time - start_time
        memory_usage = process.memory_info().rss

        best_routes = self.translate_positions_to_routes(best)
        performance_metrics = {
            "Computation Time (seconds)": computation_time,
            "Memory Usage (KB)": memory_usage,
        }

        logging.info("Optimization completed.")
        logging.info(f"Performance Metrics: {performance_metrics}")

        return best.fitness.values, best_routes, performance_metrics, best_visited_stations

    def translate_positions_to_routes(self, best_particle):
        # Initialize dictionary to hold routes for each truck
        truck_routes = {truck: [] for truck in range(self.num_trucks)}

        # Assign locations to trucks based on particle values
        for location_index, truck in enumerate(best_particle):
            if truck < self.num_trucks:
                truck_routes[truck].append(self.locations[location_index])

        # Add warehouse as the start and end point for each truck's route
        for truck in truck_routes:
            truck_routes[truck] = [self.warehouse] + truck_routes[truck] + [self.warehouse]

        return truck_routes
        
    def split_routes(self, particle):
        truck_routes = {truck: [] for truck in range(self.num_trucks)}
        for location_index, truck in enumerate(particle):
            if truck < self.num_trucks:
                truck_routes[truck].append(self.locations[location_index])
        return truck_routes

    def route_fitness(self, particle):
        total_time = 0
        total_cost = 0
        visited_locations = set()
        all_truck_visited_stations = {}

        # Split routes among trucks based on particle values
        truck_routes = {truck: [] for truck in range(self.num_trucks)}
        for loc_index, truck in enumerate(particle):
            if loc_index in visited_locations:
                # Penalize for duplicate assignments
                total_cost += 1000  # Adjust penalty as needed
            else:
                truck_routes[truck].append(self.locations[loc_index])
                visited_locations.add(loc_index)

        # Penalize for unassigned locations
        if len(visited_locations) < len(self.locations):
            total_cost += 1000 * (len(self.locations) - len(visited_locations))  # Adjust penalty as needed

        # Calculate cost for each truck's route
        for truck_id, route in truck_routes.items():
            route_with_warehouse = [self.warehouse] + route + [self.warehouse]
            route_cost, route_time, visited_stations = self.calculate_route_cost(route_with_warehouse, self.charging_stations, self.temperature)
            total_cost += route_cost
            total_time += route_time
            all_truck_visited_stations[truck_id] = visited_stations  # Store visited stations

        return total_cost, all_truck_visited_stations

    def calculate_distance_and_time(self, start, end):
        # Retrieve distance and time using Google Maps Directions API
        cache_key = (self.location_to_key(start), self.location_to_key(end))
        if cache_key in global_distance_cache:
            distance, time = global_distance_cache[cache_key]  # Retrieve both distance and time
            logging.info(f"Using cached result for distance calculation from {start} to {end}.")
        else:
            try:
                directions_result = gmaps.directions(start, end, mode="driving")
                distance = directions_result[0]['legs'][0]['distance']['value'] / 1000.0  # km
                time = directions_result[0]['legs'][0]['duration']['value'] / 60.0  # min
                global_distance_cache[cache_key] = (distance, time)  # Store both distance and time
                logging.info(f"Calculated distance using API from {start} to {end}.")
            except Exception as e:
                logging.error(f"Error calculating distance and time: {e}")
                return float('inf'), 0
            
        return distance, time
    
    def get_traffic_delay(self, start, end):
        cache_key = (start, end)
        if cache_key in global_distance_cache:
            logging.info(f"Using cached traffic delay for {start} to {end}.")
            return global_distance_cache[cache_key]

        now = int(current_time())
        try:
            directions_result = gmaps.directions(start, end, mode="driving", departure_time=now, traffic_model='best_guess')
            duration_in_traffic = directions_result[0]['legs'][0]['duration_in_traffic']['value']
            normal_duration = directions_result[0]['legs'][0]['duration']['value']
            traffic_delay = max(0, duration_in_traffic - normal_duration) / 60.0  # Convert seconds to minutes
            global_distance_cache[cache_key] = traffic_delay  # Cache the result
            return global_distance_cache[cache_key]
        except Exception as e:
            logging.error(f"Error calculating traffic delay: {e}")
            return 0
    
    def format_output(self, best_routes, all_visited_stations):
        output = {}
        for truck_id, route_coords in best_routes.items():
            truck_label = f"Truck {truck_id + 1}"
            route_cost, route_time, _ = self.calculate_route_cost(route_coords, self.charging_stations, self.temperature)
            route_summary = {
                'route': ' -> '.join(route_coords),
                'visited_charging_stations': all_visited_stations[truck_id],  # Include visited charging stations
                'total travel time': round(route_time, 1),
                'distance': round(route_cost, 1)
            }
            output[truck_label] = route_summary

        return output