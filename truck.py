class Truck:
    truck_id_counter = 0

    def _init_(self, capacity_kWh=400, consumption_rate_kWh_per_km=1.2, battery_level=100):
        self.truck_id = Truck.truck_id_counter
        Truck.truck_id_counter += 1
        self.capacity_kWh = capacity_kWh  # Average capacity of an EV truck
        self.consumption_rate_kWh_per_km = consumption_rate_kWh_per_km  # Average consumption rate
        self.battery_level = battery_level  # As a percentage of full capacity
    
    def update_battery_level(self, distance_km, temperature):
        # Adjusting efficiency based on temperature
        efficiency_factor = 1.2 if temperature < -10 else 1

        used_kWh = distance_km * self.consumption_rate_kWh_per_km * efficiency_factor
        self.battery_level -= (used_kWh / self.capacity_kWh) * 100

        if self.battery_level < 0:
            self.battery_level = 0

    def calculate_kWh_used(self, distance_km, temperature):
        efficiency_factor = 1.2 if temperature < -10 else 1
        used_kWh = distance_km * self.consumption_rate_kWh_per_km * efficiency_factor
        return used_kWh
    
    def needs_charging(self):
        # Trucks typically start considering charging when they reach about 20-30% battery level
        return self.battery_level <= 25
    
    def can_reach(self, distance_km):
        # Checking if the truck can reach the next destination with its current battery level
        required_kWh = distance_km * self.consumption_rate_kWh_per_km
        required_battery_level = (required_kWh / self.capacity_kWh) * 100
        return self.battery_level >= required_battery_level

    def project_battery_level_after_distance(self, distance_km, temperature):
        efficiency_factor = 1.2 if temperature < -10 else 1
        used_kWh = distance_km * self.consumption_rate_kWh_per_km * efficiency_factor
        projected_battery_level = self.battery_level - (used_kWh / self.capacity_kWh) * 100
        return max(0, projected_battery_level)  # Ensure battery level doesn't go negative
    
    def can_reach_with_min_battery(self, distance_km, min_battery_percentage=15):
        required_kWh = distance_km * self.consumption_rate_kWh_per_km
        required_battery_level = (required_kWh / self.capacity_kWh) * 100
        return self.battery_level - required_battery_level >= min_battery_percentage
    
    def charge_battery(self):
        # Simulating a full charge
        self.battery_level = 100
    
    def get_current_location(self, current_index, route, geocoded_locations):
        location_name = route[current_index] if current_index < len(route) else None
        return geocoded_locations.get(location_name, None)
    
    def calculate_required_battery_level(self, distance_km):
        """
        Calculate the required battery level to travel a given distance.
        """
        required_kWh = distance_km * self.consumption_rate_kWh_per_km
        required_battery_percentage = (required_kWh / self.capacity_kWh) * 100
        return required_battery_percentage