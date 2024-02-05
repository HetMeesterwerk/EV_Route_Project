import random
import networkx as nx
import googlemaps
import logging
import math
import time
import psutil
from truck import Truck
from global_cache import global_distance_cache

gmaps = googlemaps.Client(key='Google Key Blocked')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AntColonyOptimization:
    def _init_(self, warehouse, locations, charging_stations, num_trucks, iterations, temperature, consider_traffic=True, alpha=1, beta=6, rho=0.5, q0=0.85, initial_pheromone=0.1):
        self.warehouse = warehouse
        self.locations = locations
        self.charging_stations = charging_stations
        self.num_trucks = num_trucks
        self.iterations = iterations
        self.temperature = temperature
        self.consider_traffic = consider_traffic
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.api_call_count = 0
        self.graph = None
        self.max_pheromone = 10.0
        self.min_pheromone = 0.1
        logging.info(f"Initialized Ant Colony Optimization with {num_trucks} trucks, {len(locations)} locations, and {iterations} iterations.")

    def create_graph(self):
        """
        Create a graph representing the locations with edges 
        weighted by the road distance between locations.
        """
        geocoded_locations = self.geocode_locations()
        self.graph = nx.Graph()
        for start in geocoded_locations:
            for end in geocoded_locations:
                if start != end:
                    distance = self.calculate_road_distance(geocoded_locations[start], geocoded_locations[end])
                    if self.consider_traffic:
                        traffic_delay = self.get_traffic_delay(geocoded_locations[start], geocoded_locations[end])
                        distance += traffic_delay
                    self.graph.add_edge(start, end, weight=distance)
        logging.info("Graph creation completed.")
        return self.graph

    def haversine_distance(self, lat1, lng1, lat2, lng2):
        # Calculate Haversine distance using latitude and longitude
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
                distance = self.haversine_distance(current_lat, current_lng, station_key[0], station_key[1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_station = station

            if nearest_station is None:
                raise ValueError("No nearby charging station found.")

            return nearest_station, min_distance
        except Exception as e:
            logging.error(f"Error in find_nearest_charging_station from {current_location}: {e}")
            return None, float('inf')
        
    def calculate_distance_and_time(self, start, end, truck):
        # Check if the truck needs to visit a charging station
        distance_to_next = self.calculate_road_distance(start, end)
        logging.debug(f"Distance to next (calculate_distance_and_time): {distance_to_next}, Type: {type(distance_to_next)}")
        visited_charging_stations = []  # Added line to track visited charging stations

        if not truck.can_reach_with_min_battery(distance_to_next, min_battery_percentage=15):
            # Find nearest charging station from start location
            nearest_station, distance_to_station = self.find_nearest_charging_station(start, self.charging_stations)
            if nearest_station:
                # Update distance to include detour to charging station
                distance_to_next += distance_to_station
                distance_to_next += self.calculate_road_distance(nearest_station, end)
                truck.charge_battery()  # Simulate charging the truck
                visited_charging_stations.append(nearest_station)  # Added line to record the charging station
            else:
                logging.error("No nearby charging station found. Unable to charge the truck.")
        return distance_to_next, visited_charging_stations  # Modified return statement to include visited stations

    def calculate_road_distance(self, start, end):
        """
        Calculate the road distance between two locations.
        If available, use cached data to minimize API calls.
        """
        cache_key = (start, end)
        if cache_key in global_distance_cache:
            logging.info(f"Using cached road distance for distance from {start} to {end}.")
            return global_distance_cache[cache_key]

        try:
            self.api_call_count += 1
            directions_result = gmaps.directions(start, end, mode="driving")
            distance = directions_result[0]['legs'][0]['distance']['value'] / 1000.0  # Convert meters to kilometers
            logging.info(f"Calculating distance from {start} to {end}: {distance}")
            global_distance_cache[cache_key] = distance
            return distance
        except Exception as e:
            logging.error(f"Error calculating road distance from {start} to {end}: {e}")
            return float('inf')

    def get_traffic_delay(self, start, end):
        cache_key = (start, end)
        if cache_key in global_distance_cache:
            logging.info(f"Using cached traffic delay for {start} to {end}.")
            return global_distance_cache[cache_key]
        
        try:
            directions_result = gmaps.directions(start, end, mode="driving", departure_time='now', traffic_model='best_guess') #morning rush hour @ 7am - 10am, evening rush hour @ 4pm - 7pm
            self.api_call_count += 1
            traffic_delay = directions_result[0]['legs'][0]['duration_in_traffic']['value'] - directions_result[0]['legs'][0]['duration']['value']
            logging.info(f"Calculating traffic delay from {start} to {end}. Delay: {traffic_delay}")
            global_distance_cache[cache_key] = traffic_delay
            return global_distance_cache[cache_key]
        
        except Exception as e:
            # Handle exceptions
            logging.error(f"Error calculating traffic delay: {e}")
            return 0  # Assume no delay in case of failure

    def initialize_pheromone_trails(self):
        G = self.create_graph()
        pheromones = {tuple(sorted(edge)): self.initial_pheromone for edge in G.edges}
        return pheromones

    def construct_solutions(self, G, pheromones, temperature, q0=0.9):
        logging.info("Constructing solutions...")
        remaining_locations = set(self.locations)
        solutions = []

        for truck_index in range(self.num_trucks):
            truck = Truck(capacity_kWh=300, consumption_rate_kWh_per_km=0.2)
            solution = [self.warehouse]
            visited = set()

            while remaining_locations - visited:
                current_node = solution[-1]

                if current_node not in G:
                    logging.error(f"Node {current_node} not found in graph.")
                    continue

                neighbors = [n for n in G.neighbors(current_node) if n in remaining_locations]

                if not neighbors:
                    break  # No more neighbors to visit

                probabilities = self.calculate_transition_probabilities(current_node, neighbors, pheromones, G)

                if random.random() < q0:
                    max_probability = max(probabilities)
                    next_node = neighbors[probabilities.index(max_probability)]
                else:
                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]

                # Calculate distance to next node, considering charging stations if needed
                distance_to_next, _ = self.calculate_distance_and_time(current_node, next_node, truck)
                logging.debug(f"Distance to next in construct_solutions: {distance_to_next}, Type: {type(distance_to_next)}")
                if distance_to_next == float('inf'):
                    break  # Unreachable destination, break the loop

                truck.update_battery_level(distance_to_next, temperature)
                logging.debug(f"Updating battery level with distance: {distance_to_next}, Type: {type(distance_to_next)}")
                solution.append(next_node)
                visited.add(next_node)
                remaining_locations.discard(next_node)

            solution.append(self.warehouse)  # Return to warehouse
            logging.info(f"Solution for Truck {truck_index} completed: {solution}.")
            solutions.append(solution)

        logging.info("Solution construction completed for all trucks.")
        return solutions
    
    def calculate_truck_route_cost(self, start, end, truck):
        """
        Calculate the route cost for a truck, considering battery level and 
        the need for charging stations.
        """
        distance_to_next = self.calculate_road_distance(start, end)
        visited_charging_stations = []

        if not truck.can_reach_with_min_battery(distance_to_next, min_battery_percentage=15):
            nearest_station, distance_to_station = self.find_nearest_charging_station(start, self.charging_stations)
            if nearest_station:
                distance_to_next += distance_to_station
                distance_to_next += self.calculate_road_distance(nearest_station, end)
                truck.charge_battery()
                visited_charging_stations.append(nearest_station)
            else:
                logging.error("No nearby charging station found. Unable to charge the truck.")
        return distance_to_next, visited_charging_stations

    def calculate_transition_probabilities(self, current_node, neighbors, pheromones, G):
            pheromone_list = []
            heuristic_list = []
            for neighbor in neighbors:
                pheromone = pheromones.get(tuple(sorted((current_node, neighbor))), 0)
                distance = G[current_node][neighbor]['weight']  # Ensure this is a distance value, not a tuple
                if distance == 0:
                    distance = 0.0001
                heuristic = 1 / distance
                pheromone_list.append(pheromone)
                heuristic_list.append(heuristic)
            probabilities = [
                (pher * self.alpha) * (heur * self.beta)
                for pher, heur in zip(pheromone_list, heuristic_list)
            ]
            total = sum(probabilities)
            return [p / total for p in probabilities] if total > 0 else [1 / len(neighbors)] * len(neighbors)
    
    def find_iteration_best(self, solutions):
        best_solution = None
        best_cost = float('inf')
        best_visited_stations = []

        for solution in solutions:
            cost, stations = self.calculate_cost(solution)
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
                best_visited_stations = stations

        return best_solution, best_cost, best_visited_stations

    def pheromone_update(self, pheromones, solutions):
        # Apply pheromone decay and update
        for edge in pheromones:
            pheromones[edge] *= (1 - self.rho)
            pheromones[edge] = max(min(pheromones[edge], self.max_pheromone), self.min_pheromone)

        # Update based on all trucks
        for solution in solutions:
            for i in range(len(solution) - 1):
                start, end = solution[i], solution[i + 1]
                edge = tuple(sorted((start, end)))
                if edge in pheromones:
                    cost, _ = self.calculate_cost(solution)  # Extract only the cost
                    pheromones[edge] += 1.0 / cost
                    pheromones[edge] = max(min(pheromones[edge], self.max_pheromone), self.min_pheromone)

        return pheromones
    
    def calculate_cost(self, solution):
        total_cost = 0
        truck = Truck(capacity_kWh=300, consumption_rate_kWh_per_km=0.2)
        visited_charging_stations = []  # List to track all visited charging stations for this solution

        for i in range(len(solution) - 1):
            start, end = solution[i], solution[i + 1]
            distance, stations = self.calculate_distance_and_time(start, end, truck)

            if distance == float('inf'):
                return float('inf'), []  # Unreachable destination, return 'inf' cost and empty list for stations

            total_cost += distance
            visited_charging_stations.extend(stations)  # Append visited stations

        return total_cost, visited_charging_stations  # Return total cost and all visited stations

    def geocode_locations(self):
        geocoded_locations = {}
        for location in [self.warehouse] + self.locations:
            geocode_result = gmaps.geocode(location)
            if geocode_result:
                lat_lng = f"{geocode_result[0]['geometry']['location']['lat']},{geocode_result[0]['geometry']['location']['lng']}"
                geocoded_locations[location] = lat_lng
            else:
                logging.error(f"Geocoding failed for location: {location}")
        return geocoded_locations
    
    def optimize_routes(self):
        """
        Perform the optimization process using Ant Colony Optimization.
        """
        logging.info("Starting optimization...")
        start_time = time.time()
        process = psutil.Process()

        self.create_graph()
        pheromones = self.initialize_pheromone_trails()

        best_solutions_global = None
        best_cost_global = float('inf')
        best_visited_stations_global = []

        for t in range(self.iterations):
            solutions = self.construct_solutions(self.graph, pheromones, self.temperature)
            iteration_best_solution, iteration_best_cost, _ = self.find_iteration_best(solutions)

            if iteration_best_cost < best_cost_global:
                best_solutions_global = iteration_best_solution
                best_cost_global = iteration_best_cost
                best_visited_stations_global = [self.calculate_cost(solution)[1] for solution in iteration_best_solution]

            pheromones = self.pheromone_update(pheromones, solutions)

        end_time = time.time()
        computation_time = end_time - start_time
        memory_usage = process.memory_info().rss

        best_routes = self.format_output(best_solutions_global, best_cost_global, best_visited_stations_global)

        performance_metrics = {
            "Computation Time (seconds)": computation_time,
            "Memory Usage (KB)": memory_usage,
        }

        return best_routes, performance_metrics

    def format_output(self, best_solutions, best_cost_global, best_visited_stations):
        """
        Format the output of the optimization process for better readability.
        """
        output = {}
        for truck_id, (route, visited_stations) in enumerate(zip(best_solutions, best_visited_stations)):
            truck_label = f"Truck {truck_id + 1}"
            route_cost = self.calculate_cost(route)[0]
            route_summary = self.calculate_route_metrics(route)

            output[truck_label] = {
                'route': ' -> '.join(route),
                'visited_charging_stations': visited_stations,
                'distance (km)': route_summary['total_distance'],
                'total travel time (min)': route_summary['total_time'],
                'total kWh used': route_summary['total_kWh']
            }

        output['total_combined_cost'] = best_cost_global
        return output
    
    def location_to_key(self, location):
        if isinstance(location, dict):
            # Assuming the dictionary has 'lat' and 'lng' as keys
            return (location['lat'], location['lng'])
        return location  # If it's already a suitable type
    
    def calculate_route_metrics(self, solution):
        total_distance = 0
        total_time = 0
        total_kWh = 0
        truck = Truck(capacity_kWh=300, consumption_rate_kWh_per_km=0.2)

        for i in range(len(solution) - 1):
            start, end = solution[i], solution[i + 1]

            # Calculate distance and time to next node, considering charging stations if needed
            distance, time = self.calculate_distance_and_time(start, end, truck)

            if distance == float('inf'):
                return {'total_distance': float('inf'), 'total_time': float('inf'), 'total_kWh': float('inf')}  # Unreachable destination

            total_distance += distance
            total_time += time
            total_kWh += truck.calculate_kWh_used(distance, self.temperature)

        return total_distance, total_time, total_kWh

    def extract_route_from_solution(self, solution):
        if solution[0] != self.warehouse:
            solution.insert(0, self.warehouse)
        if solution[-1] != self.warehouse:
            solution.append(self.warehouse)
        return solution