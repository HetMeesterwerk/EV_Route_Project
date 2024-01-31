from cachetools import cached, TTLCache
from ratelimit import limits, sleep_and_retry
import googlemaps
import logging
import random
import math
import time
import psutil
from ratelimit import limits, RateLimitException
from time import time as current_time
from truck import Truck
from utils import assign_customers_to_trucks
from global_cache import global_distance_cache

gmaps = googlemaps.Client(key='Google Key Blocked')

class GeneticAlgorithm:
    def _init_(self, warehouse, locations, charging_stations, temperature, population_size, num_trucks, generations, consider_traffic=True, crossover_rate=0.8, mutation_rate=0.05, tournament_size=3):
        self.warehouse = warehouse
        self.locations = locations
        self.charging_stations = charging_stations
        self.temperature = temperature
        self.population_size = population_size
        self.num_trucks = num_trucks
        self.generations = generations
        self.consider_traffic = consider_traffic
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.api_call_counter = 0

    def generate_initial_routes(self):
        initial_population = []
        for _ in range(self.population_size):
            truck_assignments = assign_customers_to_trucks(self.num_trucks, self.locations.copy())
            individual_routes = {}
            for truck_id, truck_locations in truck_assignments.items():
                random.shuffle(truck_locations)
                individual_routes[truck_id] = [self.warehouse] + truck_locations + [self.warehouse]
            initial_population.append(individual_routes)
        return initial_population

    def calculate_distance_and_time(self, start, end):
        # Retrieve distance and time using Google Maps Directions API
        cache_key = (self.location_to_key(start), self.location_to_key(end))
        if cache_key in global_distance_cache:
            distance, time = global_distance_cache[cache_key]  # Retrieve both distance and time
            logging.info(f"Using cached result for distance calculation from {start} to {end}.")
        else:
            try:
                self.api_call_counter += 1
                directions_result = gmaps.directions(start, end, mode="driving")
                distance = directions_result[0]['legs'][0]['distance']['value'] / 1000.0  # km
                time = directions_result[0]['legs'][0]['duration']['value'] / 60.0  # min
                global_distance_cache[cache_key] = (distance, time)  # Store both distance and time
                logging.info(f"Calculated distance using API from {start} to {end}.")
            except Exception as e:
                logging.error(f"Error calculating distance and time: {e}")
                return float('inf'), 0
            
        return distance, time

    def can_proceed_to_next_with_charging_option(self, truck, current_location, next_location, charging_stations, temperature):
        """
        Check if the truck can proceed to the next location and then reach a charging station if needed.
        """
        logging.info("Checking if truck can proceed to next with charging option.")
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

    def location_to_key(self, location):
        if isinstance(location, dict):
            # Assuming the dictionary has 'lat' and 'lng' as keys
            return (location['lat'], location['lng'])
        return location  # If it's already a suitable type
        
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

    def format_output(self, best_routes):
        output = {}
        for truck_id, route in best_routes.items():
            distance, time, kWh = self.calculate_route_metrics(route)
            route_summary = {
                'route': ' -> '.join(route),
                'distance': distance,
                'total travel time': time,
                'total kWh used': kWh
            }
            output[f"Truck {truck_id}"] = route_summary

        # Adding combined metrics for all trucks
        total_distance, total_time, total_kWh = self.calculate_combined_metrics(best_routes)
        output['Combined Metrics'] = {
            'total distance': total_distance,
            'total time': total_time,
            'total kWh': total_kWh
        }
        logging.info(f"Output formatted: {output}")
        return output
    
    def calculate_combined_metrics(self, routes):
        total_distance, total_time, total_kWh = 0, 0, 0
        for route in routes.values():
            dist, time, kWh = self.calculate_route_metrics(route)
            total_distance += dist
            total_time += time
            total_kWh += kWh
        return total_distance, total_time, total_kWh
    
    def tournament_selection(self, population, fitness_scores):
        selected_population = []
        while len(selected_population) < len(population) // 2:
            tournament = random.sample(list(zip(population, fitness_scores)), self.tournament_size)
            winner = min(tournament, key=lambda x: x[1])[0]  # Selecting the one with the lowest cost
            selected_population.append(winner)
        return selected_population
    
    def euclidean_distance(self, lat1, lng1, lat2, lng2):
        # Calculate Euclidean distance using latitude and longitude
        R = 6371  # Radius of the Earth in km
        dLat = math.radians(lat2 - lat1)
        dLng = math.radians(lng2 - lng1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLng/2) * math.sin(dLng/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance
    
    def crossover(self, parent1, parent2):
        child1, child2 = {}, {}
        for truck_id in parent1:
            if random.random() < self.crossover_rate:
                # Perform crossover, excluding the first and last elements (warehouse)
                size = len(parent1[truck_id]) - 2  # Excluding warehouse
                if size > 2:
                    idx1, idx2 = sorted(random.sample(range(1, size), 2))
                    midsection1 = parent1[truck_id][idx1:idx2]
                    midsection2 = parent2[truck_id][idx1:idx2]
                    remaining1 = [loc for loc in parent1[truck_id][1:-1] if loc not in midsection2]
                    remaining2 = [loc for loc in parent2[truck_id][1:-1] if loc not in midsection1]
                    child1[truck_id] = [self.warehouse] + remaining1[:idx1 - 1] + midsection2 + remaining1[idx1 - 1:] + [self.warehouse]
                    child2[truck_id] = [self.warehouse] + remaining2[:idx1 - 1] + midsection1 + remaining2[idx1 - 1:] + [self.warehouse]
                else:
                    child1[truck_id], child2[truck_id] = parent1[truck_id][:], parent2[truck_id][:]
            else:
                child1[truck_id], child2[truck_id] = parent1[truck_id][:], parent2[truck_id][:]
        return child1, child2

    def mutate(self, route_set):
        mutated_set = {}
        for truck_id, route in route_set.items():
            mutated_route = route[:]
            truck = Truck(capacity_kWh=300, consumption_rate_kWh_per_km=0.2)
            for i in range(1, len(mutated_route) - 1):
                if random.random() < self.mutation_rate:
                    idx = random.randint(1, len(mutated_route) - 2)
                    mutated_route[i], mutated_route[idx] = mutated_route[idx], mutated_route[i]
                    # After mutation, check if the route is still feasible
                    calculated_route, visited_charging_stations = self.calculate_route_cost({truck_id: mutated_route}, self.charging_stations)
                    if calculated_route == float('inf'):
                        # If not feasible, revert the mutation
                        mutated_route[i], mutated_route[idx] = mutated_route[idx], mutated_route[i]
            mutated_set[truck_id] = mutated_route
        return mutated_set

    def evaluate_population(self, population):
        fitness_scores = []
        for individual in population:
            total_cost, _ = self.calculate_route_cost(individual, self.charging_stations)
            fitness_scores.append(total_cost)
        return fitness_scores
        
    def calculate_route_cost(self, routes, charging_stations):
        total_cost = 0
        visited_charging_stations = []
        for truck_id, route in routes.items():
            truck_cost = 0
            truck = Truck(capacity_kWh=300, consumption_rate_kWh_per_km=0.2)
            logging.info(f"Calculating route cost for Truck {truck_id}")

            for i in range(len(route) - 1):
                start, end = route[i], route[i + 1]
                can_proceed, nearest_station, distance_to_station = self.can_proceed_to_next_with_charging_option(truck, start, end, charging_stations, self.temperature)

                if can_proceed:
                    distance_to_next_location, _= self.calculate_distance_and_time(start, end)
                    truck_cost += distance_to_next_location
                    truck.update_battery_level(distance_to_next_location, self.temperature)
                    if nearest_station:
                        truck_cost += distance_to_station
                        truck.charge_battery()
                        visited_charging_stations.append(nearest_station)
                else:
                    # Divert to nearest charging station from current location
                    nearest_station_from_current, distance_to_station_from_current = self.find_nearest_charging_station(start, charging_stations)
                    if nearest_station_from_current is not None:
                        truck_cost += distance_to_station_from_current
                        truck.charge_battery()
                        visited_charging_stations.append(nearest_station_from_current)
                        additional_distance_to_next, _ = self.calculate_distance_and_time(nearest_station_from_current, end)
                        truck_cost += additional_distance_to_next
                        truck.update_battery_level(additional_distance_to_next, self.temperature)
                    else:
                        return float('inf'), visited_charging_stations  # Infeasible route

                if self.consider_traffic:
                    delay = self.get_traffic_delay(start, end)
                    truck_cost += delay

            total_cost += truck_cost

        logging.info(f"Total route cost: {total_cost} km, visited charging stations: {visited_charging_stations}")
        return total_cost, visited_charging_stations
    
    def calculate_route_metrics(self, route):
        total_distance = 0
        total_time = 0
        total_kWh = 0
        truck = Truck(300, 0.2)  # Create a Truck object

        for i in range(len(route) - 1):
            start, end = route[i], route[i + 1]
            distance, time = self.calculate_distance_and_time(start, end)
            total_distance += distance
            total_time += time
            total_kWh += truck.calculate_kWh_used(distance, self.temperature)

        return total_distance, total_time, total_kWh
        
    def find_nearest_charging_station(self, current_location, charging_stations):
        # Convert the current location to a cache-friendly format
        current_key = self.location_to_key(current_location)
        current_lat, current_lng = global_distance_cache.get(current_key, (None, None))

        # If not cached, fetch geocode and update cache
        if current_lat is None or current_lng is None:
            try:
                geocode_result = gmaps.geocode(current_location)
                if not geocode_result:
                    raise ValueError(f"Geocoding failed for location: {current_location}")

                current_lat = geocode_result[0]['geometry']['location']['lat']
                current_lng = geocode_result[0]['geometry']['location']['lng']
                global_distance_cache[current_key] = (current_lat, current_lng)
            except Exception as e:
                logging.error(f"Error in find_nearest_charging_station: {e}")
                return None, float('inf')

        # Find the nearest charging station
        nearest_station, min_distance = None, float('inf')
        for station in charging_stations:
            station_key = self.location_to_key(station)
            distance = self.euclidean_distance(current_lat, current_lng, station_key[0], station_key[1])
            if distance < min_distance:
                min_distance = distance
                nearest_station = station

        # Return the nearest charging station and its distance
        return nearest_station, min_distance

    def run(self):
        # Initialize the population
        start_time = time.time()
        population = self.generate_initial_routes()
        process = psutil.Process()

        for generation in range(self.generations):
            # Evaluate the fitness of each individual in the population
            fitness_scores = self.evaluate_population(population)

            # Tournament Selection
            selected_population = self.tournament_selection(population, fitness_scores)

            # Crossover and Mutation
            new_population = []
            while len(new_population) < self.population_size:
                if len(selected_population) >= 2:
                    parent1, parent2 = random.sample(selected_population, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(self.mutate(child1))
                    if len(new_population) < self.population_size:
                        new_population.append(self.mutate(child2))
                else:
                    mutated_individual = self.mutate(selected_population[0])
                    new_population.append(mutated_individual)

            # Update the population
            population = new_population

        # Select the best individual from the final population
        best_individual_idx = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx])[0]
        best_routes = population[best_individual_idx]

        # Computation metrics
        end_time = time.time()
        computation_time = end_time - start_time
        memory_usage = process.memory_info().rss

        # Format and return the best routes found
        return self.format_output(best_routes), {
            "Computation Time (seconds)": computation_time,
            "Memory Usage (KB)": memory_usage
        }