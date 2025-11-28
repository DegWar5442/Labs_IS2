# **Vehicle Transport Ontology 🚗✈️🚢**

This project contains a semantic model (Ontology) defined in **OWL (Web Ontology Language)**. It formalizes the domain of transportation, defining relationships between vehicles, the infrastructure they navigate, their mechanical components, and the people who operate them.

## **📂 Ontology Details**

* **Format:** RDF/XML  
* **Base IRI:** http://www.example.org/vehicles.owl  
* **Language:** OWL 2 DL

## **🧠 Class Hierarchy**

The ontology is organized into five main branches:

### **1\. Vehicles (\#Vehicle)**

Classified by the medium in which they travel:

* **LandVehicle:** Car, Truck, Motorcycle, Bicycle.  
* **Watercraft:** Ship, Boat, Submarine.  
* **Aircraft:** Airplane, Helicopter, Glider.  
* **Spacecraft:** Vehicles designed for space travel.

### **2\. Infrastructure (\#Infrastructure)**

The environment required for vehicle operation:

* **Road:** Highway, Offroad tracks.  
* **WaterBody:** Ocean, River.  
* **Airspace** and **Space**.

### **3\. Parts (\#VehiclePart)**

Components that make up a vehicle:

* **Propulsion:** Engine (Combustion/Electric), Propeller, Sail.  
* **Structural:** Hull, Fuselage, Wing, Mast, Keel.  
* **Control:** SteeringWheel, Handlebar, Cockpit, Brake.

### **4\. People (\#Person)**

* **Operators:** Driver, Pilot, Captain.  
* **Passenger:** Non-operators.

### **5\. Operations (\#Operation)**

Actions performed by the vehicles: Drive, Fly, Sail, Dive, Orbit.

## **🔗 Key Relationships (Object Properties)**

The model uses specific properties to link classes together:

| Property | Description | Characteristics |
| :---- | :---- | :---- |
| travels\_on | Links a vehicle to its infrastructure (e.g., Car → Road). | \- |
| hasPart | Defines the composition of a vehicle. | **Transitive** (A part of a part is a part of the whole). |
| operated\_by | Links a vehicle to the correct operator type. | Inverse of operates. |
| performs | Links a vehicle to its action (e.g., Submarine → Dive). | \- |

## **🛡️ Logic & Constraints**

The ontology enforces strict logical rules to ensure data consistency:

1. **Disjoint Classes:**  
   * A LandVehicle cannot be an Aircraft or Watercraft.  
   * A Car cannot be a Truck or Motorcycle.  
   * A Wheel cannot be a Propeller.  
2. **Restrictions (Examples):**  
   * A **Car** *must* have an Engine, Wheels, and travel on a Road.  
   * A **Glider** is defined as an Aircraft that flies and has wings, but (implicitly) lacks an engine.  
   * A **Bicycle** is defined to travel on both Road and Offroad.  
3. **Inference:**  
   * Because hasPart is transitive, if an Engine has a Piston, the Car containing that Engine also contains the Piston.

## **🧪 Sample Data (Individuals)**

The file includes named individuals to demonstrate the schema:

* **\#Tesla\_Model\_S**: Defined as a Car with an ElectricMotor.  
* **\#Titanic**: Defined as a Ship.  
* **\#Spirit\_of\_St\_Louis**: Defined as an Airplane.

## **🛠 How to Use**

1. Save the content as vehicles.owl.  
2. Open the file in **Protégé** (an open-source ontology editor).  
3. Navigate to the **Entities** tab to view the class hierarchy.  
4. Run a **Reasoner** (like HermiT) to validate the logical consistency and see inferred relationships.