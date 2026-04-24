# Shahd Shwekeyeh-1222105 sec:3
# Dareen Abualhaj-1220686 sec:3

# Import required libraries
import math
import random
import time
import matplotlib.pyplot as plt   
import matplotlib                 
import numpy as np
from typing import List, Tuple    

# Class to represent a package with destination coordinates, weight, and priority
class Package:
    def __init__(self, x: float, y: float, weight: float, priority: int):
        self.x = x  # X coordinate of destination
        self.y = y  # Y coordinate of destination
        self.weight = weight  # Weight of the package in kg
        self.priority = priority  # Priority level (1 = highest priority)
    
    def __repr__(self):
        # String representation of the package for printing
        return f"Package(dest=[{self.x},{self.y}], weight={self.weight}, priority={self.priority})"

# Class to represent a delivery vehicle with capacity and assigned packages
class Vehicle:
    def __init__(self, capacity: float):
        self.capacity = capacity  # Maximum weight capacity in kg
        self.packages: List[Package] = []  # List to store assigned packages
    
    def total_weight(self) -> float:
        # Calculate total weight of all packages in the vehicle
        return sum(p.weight for p in self.packages)
    
    def add_package(self, package: Package) -> bool:
        # Add a package if it doesn't exceed vehicle capacity
        if self.total_weight() + package.weight <= self.capacity:
            self.packages.append(package)
            return True
        return False
    
    def clear_packages(self):
        # Remove all packages from the vehicle
        self.packages = []
    
    def __repr__(self):
        # String representation of the vehicle for printing
        return f"Vehicle(capacity={self.capacity}, used={self.total_weight()}, packages={len(self.packages)})"

# Calculate Euclidean distance between two points (x1,y1) and (x2,y2)
def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)

# Calculate total distance for a vehicle's route (round trip from shop)
def compute_route_distance(route: List[Package]) -> float:
    if not route:
        return 0.0  # No distance if no packages
    
    distance = 0.0
    x_prev, y_prev = 0.0, 0.0  # Start from shop location (0,0)
    
    # Add distance between consecutive package destinations
    for package in route:
        distance += euclidean_distance(x_prev, y_prev, package.x, package.y)
        x_prev, y_prev = package.x, package.y
    
    # Add return distance to shop
    distance += euclidean_distance(x_prev, y_prev, 0.0, 0.0)
    return distance

# Calculate total distance for all vehicles plus penalty for unassigned packages
def compute_total_distance(vehicles: List[Vehicle], packages: List[Package]) -> Tuple[float, int]:
    total_distance = 0.0
    assigned_packages = []
    
    # Sum distances for all vehicle routes
    for vehicle in vehicles:
        total_distance += compute_route_distance(vehicle.packages)
        assigned_packages.extend(vehicle.packages)
    
    # Identify unassigned packages and apply penalty
    unassigned = [p for p in packages if p not in assigned_packages]
    unassigned_penalty = 1000 * len(unassigned)  # Large penalty per unassigned package
    
    return total_distance + unassigned_penalty, len(unassigned)

# Visualize vehicle routes with arrows and package labels
def plot_vehicle_path(vehicles: List[Vehicle]):
    cmap = matplotlib.colormaps.get_cmap('tab20')  # Color map for different vehicles
    plt.figure(figsize=(20, 18))
    package_counter = 1  # Counter for numbering packages

    # Plot route for each vehicle
    for idx, vehicle in enumerate(vehicles):
        if not vehicle.packages:
            continue  # Skip vehicles with no packages
        
        color = cmap(idx / max(1, len(vehicles)))  # Assign unique color
        x_coords = [0] + [p.x for p in vehicle.packages] + [0]  # Include shop at start/end
        y_coords = [0] + [p.y for p in vehicle.packages] + [0]
        
        # Plot the route line
        plt.plot(x_coords, y_coords, color=color, marker='o', linestyle='-', 
                label=f'Vehicle #{idx+1} ({len(vehicle.packages)} packages)', linewidth=2)
        
        # Add directional arrows between points
        for i in range(len(x_coords) - 1):
            arrow_color = 'red' if i == len(x_coords) - 2 else color  # Red for return to shop
            plt.annotate('', xy=(x_coords[i+1], y_coords[i+1]), xytext=(x_coords[i], y_coords[i]),
                         arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2 if i == len(x_coords)-2 else 3))
        
        # Label each package with a number
        for package in vehicle.packages:
            plt.text(package.x, package.y, f'{package_counter}', fontsize=10,
                     ha='center', va='center', color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))
            package_counter += 1
    
    # Mark and label the shop location
    plt.scatter(0, 0, color='black', s=400, marker='*', label='Shop (0,0)')
    plt.text(0, 0, 'Shop', fontsize=14, fontweight='bold', ha='left', va='bottom')
    
    # Configure plot appearance
    plt.title('Vehicle Delivery Routes', fontsize=22)
    plt.xlabel('X Coordinate (km)', fontsize=18)
    plt.ylabel('Y Coordinate (km)', fontsize=18)
    plt.legend(fontsize=12, loc='best', ncol=3)  # Multi-column legend
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Print assignment results including unassigned packages
def print_assignment_summary(vehicles: List[Vehicle], packages: List[Package]):
    assigned_packages = []
    for vehicle in vehicles:
        assigned_packages.extend(vehicle.packages)
    
    unassigned = [p for p in packages if p not in assigned_packages]
    total_distance = sum(compute_route_distance(v.packages) for v in vehicles)
    
    print("\n=== Assignment Summary ===")
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Unassigned packages: {len(unassigned)}")
    
    # Print details for each vehicle
    print("\n--- Vehicle Details ---")
    for idx, vehicle in enumerate(vehicles):
        print(f"\nVehicle #{idx+1} (Capacity: {vehicle.total_weight():.2f}/{vehicle.capacity} kg):")
        for p in vehicle.packages:
            print(f"  - {p}")
    
    # Print unassigned packages if any
    if unassigned:
        print("\n--- Unassigned Packages ---")
        for p in unassigned:
            print(f"  - {p}")

#--------------------------------------------------------------------------------------------------------------------            
# Simulated Annealing algorithm implementation
#---------------------------------------------------------------------------------------------------------------------
def simulated_annealing(vehicles: List[Vehicle], packages: List[Package]):
    # SA parameters
    initial_temp = 1000  # Starting temperature
    cooling_rate = 0.95  # Cooling rate (between 0.9-0.99)
    stopping_temp = 1    # Minimum temperature
    iterations_per_temp = 100  # Iterations at each temperature
    
    # Create initial feasible solution
    def initial_solution() -> List[Vehicle]:
        new_vehicles = [Vehicle(v.capacity) for v in vehicles]
        # Sort packages by priority (ascending) and weight (descending)
        sorted_packages = sorted(packages, key=lambda p: (p.priority, -p.weight))
        
        # Assign packages using best-fit approach
        for package in sorted_packages:
            # Try vehicles with most remaining capacity first
            sorted_vehicles = sorted(new_vehicles, 
                                    key=lambda v: v.capacity - v.total_weight(), 
                                    reverse=True)
            for vehicle in sorted_vehicles:
                if vehicle.add_package(package):
                    break
        return new_vehicles
    
    # Generate neighboring solution
    def generate_neighbor(current: List[Vehicle]) -> List[Vehicle]:
        neighbor = [Vehicle(v.capacity) for v in vehicles]
        # Copy current solution
        for i, v in enumerate(current):
            neighbor[i].packages = v.packages.copy()
        
        # Attempt package moves between vehicles
        for _ in range(3):  # Try up to 3 moves
            from_idx = random.randint(0, len(neighbor) - 1)
            if not neighbor[from_idx].packages:
                continue
                
            # Select random package to move
            pkg_idx = random.randint(0, len(neighbor[from_idx].packages) - 1)
            package = neighbor[from_idx].packages.pop(pkg_idx)
            
            # Try to assign to another random vehicle
            for to_idx in random.sample(range(len(neighbor)), len(neighbor)):
                if to_idx == from_idx:
                    continue
                if neighbor[to_idx].add_package(package):
                    return neighbor  # Successfully moved
            
            # If couldn't move, return package to original vehicle
            neighbor[from_idx].packages.append(package)
        
        return neighbor  # Return modified or original solution
    
    # Initialize current and best solutions
    current_solution = initial_solution()
    current_cost, current_unassigned = compute_total_distance(current_solution, packages)
    best_solution = [Vehicle(v.capacity) for v in vehicles]
    for i, v in enumerate(current_solution):
        best_solution[i].packages = v.packages.copy()
    best_cost = current_cost
    
    temp = initial_temp
    iteration = 0
    
    print("\nStarting Simulated Annealing...")
    print(f"Initial solution cost: {current_cost:.2f} ({current_unassigned} unassigned)")
    
    # Main SA loop
    while temp > stopping_temp:
        for _ in range(iterations_per_temp):
            neighbor = generate_neighbor(current_solution)
            neighbor_cost, neighbor_unassigned = compute_total_distance(neighbor, packages)
            delta = neighbor_cost - current_cost
            
            # Accept neighbor if better or with probability based on temperature
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_solution = neighbor
                current_cost = neighbor_cost
                current_unassigned = neighbor_unassigned
                
                # Update best solution if improved
                if neighbor_cost < best_cost:
                    best_solution = neighbor
                    best_cost = neighbor_cost
                    print(f"Iter {iteration}: New best cost: {best_cost:.2f} ({neighbor_unassigned} unassigned)")
        
        temp *= cooling_rate  # Cool down
        iteration += 1
    
    # Print final results
    print("\n=== Simulated Annealing Results ===")
    print(f"Final cost: {best_cost:.2f}")
    print_assignment_summary(best_solution, packages)
    plot_vehicle_path(best_solution)
    
    return best_solution

#---------------------------------------------------------------------------------------------------------------
# Genetic Algorithm implementation
#----------------------------------------------------------------------------------------------------------------
def genetic_algorithm(vehicles: List[Vehicle], packages: List[Package]):
    # GA parameters
    population_size = 50  # Number of solutions in population
    mutation_rate = 0.05  # Probability of mutation
    generations = 500     # Maximum generations
    elite_size = 5        # Number of elite solutions preserved
    
    # Create an individual solution
    def create_individual() -> List[Vehicle]:
        individual = [Vehicle(v.capacity) for v in vehicles]
        # Sort packages by priority and weight
        sorted_packages = sorted(packages, key=lambda p: (p.priority, -p.weight))
        
        # Assign packages using best-fit approach
        for package in sorted_packages:
            # Try vehicles with most remaining capacity first
            sorted_vehicles = sorted(individual, 
                                    key=lambda v: v.capacity - v.total_weight(), 
                                    reverse=True)
            for vehicle in sorted_vehicles:
                if vehicle.add_package(package):
                    break
        return individual
    
    # Crossover two parent solutions
    def crossover(parent1: List[Vehicle], parent2: List[Vehicle]) -> List[Vehicle]:
        child = [Vehicle(v.capacity) for v in vehicles]
        # Combine packages from both parents
        all_packages = []
        for v in parent1:
            all_packages.extend(v.packages)
        for v in parent2:
            all_packages.extend(v.packages)
        all_packages = list(set(all_packages))  # Remove duplicates
        
        # Assign packages to child using best-fit
        for package in all_packages:
            sorted_vehicles = sorted(child, 
                                    key=lambda v: v.capacity - v.total_weight(), 
                                    reverse=True)
            for vehicle in sorted_vehicles:
                if vehicle.add_package(package):
                    break
        return child
    
    # Mutate an individual solution
    def mutate(individual: List[Vehicle]):
        for _ in range(int(mutation_rate * len(packages))):
            from_idx = random.randint(0, len(individual) - 1)
            if not individual[from_idx].packages:
                continue
                
            # Select random package to move
            pkg_idx = random.randint(0, len(individual[from_idx].packages) - 1)
            package = individual[from_idx].packages.pop(pkg_idx)
            
            # Try to assign to another random vehicle
            to_idx = random.randint(0, len(individual) - 1)
            if not individual[to_idx].add_package(package):
                # Revert if couldn't assign
                individual[from_idx].packages.append(package)
    
    # Fitness function (lower is better)
    def fitness(individual: List[Vehicle]) -> float:
        cost, unassigned = compute_total_distance(individual, packages)
        return cost
    
    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    
    print("\nStarting Genetic Algorithm...")
    # Evolution loop
    for gen in range(generations):
        # Sort by fitness
        population.sort(key=fitness)
        
        # Select elite solutions to preserve
        next_generation = []
        for i in range(elite_size):
            elite = [Vehicle(v.capacity) for v in vehicles]
            for j, v in enumerate(population[i]):
                elite[j].packages = v.packages.copy()
            next_generation.append(elite)
        
        # Create offspring through crossover and mutation
        while len(next_generation) < population_size:
            # Select parents from top 20% of population
            parent1, parent2 = random.choices(population[:20], k=2)
            child = crossover(parent1, parent2)
            mutate(child)
            next_generation.append(child)
        
        population = next_generation
        
        # Print progress periodically
        if gen % 50 == 0:
            best_fitness = fitness(population[0])
            _, unassigned = compute_total_distance(population[0], packages)
            print(f"Generation {gen}: Best fitness = {best_fitness:.2f} ({unassigned} unassigned)")
    
    # Return best solution found
    best_solution = population[0]
    print("\n=== Genetic Algorithm Results ===")
    print(f"Final fitness: {fitness(best_solution):.2f}")
    print_assignment_summary(best_solution, packages)
    plot_vehicle_path(best_solution)
    
    return best_solution

# Main program execution
if __name__ == "__main__":
    print("\n===== Vehicle and Package Setup =====")
    # Get number of vehicles with input validation
    while True:
        try:
            num_vehicles = int(input("Enter the number of available vehicles: "))
            if num_vehicles > 0:
                break
            print("Please enter a positive number!")
        except ValueError:
            print("Please enter a valid integer!")

    # Get vehicle capacities with validation
    vehicles = []
    for i in range(num_vehicles):
        while True:
            try:
                cap = float(input(f"Enter capacity of vehicle #{i+1} (kg): "))
                if cap > 0:
                    break
                print("Capacity must be positive!")
            except ValueError:
                print("Please enter a valid number!")
        vehicles.append(Vehicle(capacity=cap))
    
    # Get number of packages with validation
    while True:
        try:
            num_packages = int(input("\nEnter the number of packages: "))
            if num_packages > 0:
                break
            print("Please enter a positive number!")
        except ValueError:
            print("Please enter a valid integer!")

    # Get package details with validation
    packages = []
    for i in range(num_packages):
        print(f"\nPackage #{i+1}:")
        while True:
            try:
                x = float(input("  - X coordinate: "))
                y = float(input("  - Y coordinate: "))
                weight = float(input("  - Weight (kg): "))
                if weight <= 0:
                    print("Weight must be positive!")
                    continue
                priority = int(input("  - Priority (1=highest): "))
                if priority < 1:
                    print("Priority must be at least 1!")
                    continue
                break
            except ValueError:
                print("Please enter valid numbers!")
        packages.append(Package(x, y, weight, priority))
    
    # Algorithm selection with validation
    print("\n========= Algorithm Selection =========")
    print("Choose an optimization algorithm:")
    print("1. Simulated Annealing (SA)")
    print("2. Genetic Algorithm (GA)")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("Invalid choice! Please enter 1 or 2.")
    
    # Run selected algorithm
    if choice == "1":
        simulated_annealing(vehicles, packages)
    else:
        genetic_algorithm(vehicles, packages)