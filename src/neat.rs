use core::panic;
// Based on 
// https://github.com/SirBob01/NEAT-Python/blob/master/neat/neat.py
use std::{collections::HashMap, fmt::Debug, iter};
use rand::prelude::*;

/* Current TODO
Innovation numbers - for when an edge is split/so we get the same node ID ig?
-> Have a second hasmap for when edges are split and that goes to the node id !!!
---> In "brain" need a current max please node
-> Easy for adding a new edges as just map pair to IN

===== Things that might cause errors later cus i added them late
-> Things missing from a nodes from list!!

*/

// ========== PARAMETERS ============
const PROB_MUTATE_INDIVIDUAL: f64 = 0.3;         // How often we should mutate instead of breeding 2 parents
const PROB_CHOOSE_PERFORMER: f64 = 0.75;         // How often we should take a gene from the performer
const PROB_FLIP_EDGE: f64 = 0.4;                 // How often we should deactivate/activate an edge
const PROB_ADD_NODE: f64 = 0.1;                  // How often we should split an edge
const PROB_SHIFT_BIAS: f64 = 0.2;                // How often we should shift the bias of a node
const PROB_SHIFT_WEIGHT: f64 = 0.2;              // How often we should shift the weight of an edge
const PROB_ADD_EDGE: f64 = 0.1;                  // How often we should add a new edge

const SPECIES_DELTA: f64 = 1.2;                 // The difference between 2 genomes for them to be 2 different species
const CULL_PROPORTION: f64 = 0.4;              // What proportion of a species should be culled

const EDGE_DISTANCE_WEIGHT: f64 = 1.0;          // How much we value the difference in number of weights
const BIAS_DISTANCE_WEIGHT :f64 = 0.4;          // How much we value the difference in biases
const WEIGHT_DISTANCE_WEIGHT : f64 = 1.0;       // How much we value difference in weights of the edges



fn leaky_relu(x:f64) -> f64 {
    if x > 0.0 { 
        return x;
    }
    0.01 * x
}

#[derive(Debug, Clone)]
pub struct Node {
    bias:f64,
    output:f64,
    pub from:Vec<u64>, // a vector of nodes which are connected to this node which are !!!ENABLED!!! 

    // For drawing 
    pub x: f64, // Proportions from 0 to 1 of where they should be drawn from top to bottom of the
    pub y: f64,
}

impl Node {
    fn new(x: f64, y: f64) -> Self {
        Self {
            bias:0.0,
            output:0.0,
            from:Vec::new(),
            x:x,
            y:y,
        }
    }

    /// Checks if an id is in the list of nodes which feed into us aswell as
    fn has_id(&self, id: u64) -> Option<usize>{
        for idx in 0..self.from.len(){

            if id == self.from[idx] {
                return Some(idx);
            }
        }

        None
    }
    
    /// Adds a node to the list of nodes which feed into ours
    fn add_to_from(&mut self, id: u64){
        match self.has_id(id ) {
            None => self.from.push(id),
            Some(idx) => return,
        }
    }

    /// Removes an id from the form list of nodes that feed into us
    fn remove_from_from(&mut self, id: u64){
        match self.has_id(id) {
            None => return,
            Some(idx) => self.from.remove(idx),
        };
    }
}
#[derive(Debug, Clone, Copy)]
struct Edge {
    weight:f64,
    pub enabled:bool,
}

impl Edge {
    fn new(w:f64) -> Self{
        Self {
            weight:w,
            enabled:true,
        }
    }
}
#[derive(Debug, Clone)]
pub struct Genome{
    pub inputs:u64,                     // number of inputs
    pub outputs:u64,                    // number of output nodes
    unhidden:u64,                   // number of input and output nodes
    hidden: u64,                    // The number of hidden nodes
    num_edges:u64,                  // The number of edges

    edge_list:Vec<(u64,u64)>,       // Vector of the current edges in our list so we can choose
    edges:HashMap<(u64,u64), Edge>, // map the nodes the edge is between to the actual edge
    pub nodes:HashMap<u64,Node>,        // map nodeIDs to the actual Node instance

    fitness:f64,                    // the fitness of the genome
    adjusted_fitness:f64,           //fitness but averaged within a species

    ordered_nodes:Vec<u64>,         // The ordered nodes so we can propogate without calculating at it at each step
    pub hidden_nodes:Vec<u64>,          // The ids of nodes that are not listed

    pub visited: bool,
}   

// Implementing functions for our genome
impl Genome {
    /// Create a new genome instance with the number of inputs and outputs
    pub fn new(inputs: u64, outputs:u64,)-> Self{
        Self {
            inputs:inputs,
            outputs:outputs,
            unhidden:inputs+outputs,
            hidden:0,
            num_edges:0,
            edge_list:Vec::new(),
            edges:HashMap::new(),
            nodes:HashMap::new(),
            fitness:0.0,
            adjusted_fitness:0.0,
            ordered_nodes:Vec::new(),
            hidden_nodes:Vec::new(),
            visited: false,
        }

    }  

    pub fn set_fitness(&mut self,f:f64){
        self.fitness = f;
    }

    pub fn get_fitness(&self) -> f64{
        self.fitness
    }

    /// Adds a new edge or enables it if it alraedy exists
    pub fn add_edge(&mut self, i:u64, j:u64, w:f64){
        //Check node IDs exist
        if self.nodes.contains_key(&i) == false{
            panic!("Start node id was not found")
        }

        if self.nodes.contains_key(&j) == false{
            panic!("End node id was not found")
        }

        // If the edge already exists just enable it
        if self.edges.contains_key(&(i, j)) {
            let e:&mut Edge;
            e = self.edges.get_mut(&(i,j)).unwrap();
            
            //Update params
            e.enabled = true;
            e.weight = w;
        }
        else{
            // Edge does not exist
            self.edges.insert((i,j), Edge::new(w));
            self.edge_list.push((i,j));
            self.num_edges += 1;
        }

        // Call here incase the edge was disabled before
        match self.nodes.get_mut(&j){
            None=>panic!("Tried to add to the from list of a node that is not indexed"),
            Some(node) => node.add_to_from(i),
        };

    }
    /// Disables an edge from i to j
    pub fn disable_edge(&mut self, i:u64, j:u64){
        if self.edges.contains_key(&(i, j)) {
            let e:&mut Edge;
            e = self.edges.get_mut(&(i,j)).unwrap();
            
            //Disable
            e.enabled = false;
        }else {
            panic!("Edge with invalid start and end given");
        }

        match self.nodes.get_mut(&j){
            None=>panic!("Tried to add to the from list of a node that is not indexed"),
            Some(node) => node.remove_from_from(i),
        };
    }
    /// Enables an edge from i to j
    pub fn enable_edge(&mut self, i:u64, j:u64){
        if self.edges.contains_key(&(i, j)) {
            let e:&mut Edge;
            e = self.edges.get_mut(&(i,j)).unwrap();
            
            // Enable the edge
            e.enabled = true;
            
            match self.nodes.get_mut(&j){
                None=>panic!("Node does not exist"),
                Some(node)=>node.from.push(i),
            }
        }else {
            panic!("Edge with invalid start and end given");
        }

        match self.nodes.get_mut(&j){
            None=>panic!("Tried to add to the from list of a node that is not indexed"),
            Some(node) => node.add_to_from(i),
        };
    }

    pub fn add_edge_weight(&mut self, i: u64, j: u64, w: f64){
        let e =  match self.edges.get_mut(&(i,j)){
            None=>panic!("Cannot set the weight of {}->{} does not exit",i,j),
            Some(e)=>e,
        };
        e.weight += w;
    }

    pub fn set_edge_weight(&mut self, i: u64, j: u64, w: f64){
        let e =  match self.edges.get_mut(&(i,j)){
            None=>panic!("Cannot set the weight of {}->{} does not exit",i,j),
            Some(e)=>e,
        };
        e.weight = w;
    }

    pub fn mutate_random_edge_weight(&mut self){
        // Generate random edge
        let mut rng = rand::thread_rng();
        let edge_idx = rng.gen_range(0..self.num_edges) as usize;
        let (i, j) = self.edge_list[edge_idx];
        // Generate new weight
        let w = 2.0 * (rng.gen_range(0.0..2.0) - 1.0);
        self.add_edge_weight(i, j, w);
    }

    pub fn flip_random_edge(&mut self){
        // Generate random edge
        let mut rng = rand::thread_rng();
        let edge_idx = rng.gen_range(0..self.num_edges) as usize;
        let ij = self.edge_list[edge_idx];

        let e = match self.edges.get_mut(&ij){
            None => panic!("Generated random ij that does not exist {:?}",ij),
            Some(e) => e,
        };
        e.enabled = !e.enabled;
        
    }

    /// Splits an edge to add a new node
    pub fn add_node(&mut self, i:u64, j:u64, id:u64) {
        if self.nodes.contains_key(&i) == false{
            panic!("Start node id was not found")
        }

        if self.nodes.contains_key(&j) == false{
            panic!("End node id was not found")
        }

        if self.nodes.contains_key(&id) {
            panic!("Duplicate node id attempted to be used")
        }

        // Split the edge
        self.disable_edge(i, j);
        let w = self.edges.get(&(i,j)).unwrap().weight;
        let pi = &self.nodes[&i];
        let pj = &self.nodes[&j];


        self.nodes.insert(id, Node::new(((pj.x - pi.x)/2.0).abs(), ((pj.y - pi.y)/2.0).abs() ));
        // Add new edges
        self.add_edge(i, id, 0.0);
        self.add_edge(id, j, 0.0);

        // Add to our list of hidden nodes
        self.hidden_nodes.push(id);
        self.hidden += 1;


    }

    /// Set the bias of a node
    pub fn set_node_bias(&mut self, i: u64, b: f64){
        match self.nodes.get_mut(&i) {
            None => panic!("Tried to mutate the bias of node {} does not exist",i),
            Some(n) => n.bias = b,
        };
    }

    /// Changes the bias of a random node - wont effect input nodes but that doesnt matter
    pub fn mutate_random_node_bias(&mut self){
        // Generate random node
        let mut rng = rand::thread_rng();
        let mut node_id = rng.gen_range(self.inputs..(self.unhidden+self.hidden));

        if node_id >= self.unhidden {
            node_id = self.hidden_nodes[(node_id - self.unhidden) as usize];
        }

        // Generate new weight
        let b = 2.0 * (rng.gen_range(0.0..1.0) - 0.5);
        self.set_node_bias(node_id, b);
    }

    /// Updates the ordered nodes list which is used to propogate through the network
    fn calc_ordered_nodes(&mut self) {
        // Firstly empty out our ordered nodes vector
        self.ordered_nodes.clear();

        //then want insert by their x value
        //binary search time!!!
        for i in self.hidden_nodes.iter() {
            self.ordered_nodes.push(*i);
        }



        //lazy atm should use binary search ig to insert and sort at the same time butttt
        self.ordered_nodes.sort_by(|a, b| self.nodes.get(a).unwrap().x.partial_cmp(&self.nodes.get(b).unwrap().x).unwrap_or(std::cmp::Ordering::Equal));
        
        // Add output nodes to the list
        for i in self.inputs..self.unhidden {
            self.ordered_nodes.push(i);
        }
    }

    /// Given input values feed through the network and get the outputs
    pub fn forward(&mut self,inputs:Vec<f64> ) -> Vec<f64>{
        let mut output: Vec<f64> = Vec::new();
        self.reset(); // Reset the node outputs to begin
        self.calc_ordered_nodes();

        if inputs.len() != self.inputs as usize {
            panic!("Incorrect size of inputs given {} != {}",output.len(),self.inputs)
        }

        for idx in 0..self.inputs {
            let n = self.nodes.get_mut(&idx).unwrap();
            n.output = inputs[idx as usize];
            } 
        
        let mut ix = 0.0;
        for j in self.ordered_nodes.iter() {
            
            let jx = self.nodes.get(j).unwrap().x;
            if ix > jx {
                println!("Error nodes out of order");
            }
            ix = jx;
            let mut ax: f64 = 0.0;
            if self.nodes.get(j).unwrap().from.len() == 0 {
                println!("empty from");
            }
            for i in self.nodes.get(j).unwrap().from.iter() { //loop through nodes that feed into j
                let e =  self.edges.get(&(*i, *j)).unwrap();
                ax += (self.nodes.get(i).unwrap().output * e.weight);
            }
            let node = self.nodes.get_mut(j).unwrap();
            // SET ACTIVATION HERE
            node.output = leaky_relu(ax);

        }

        // Get the output values
        for idx in self.inputs..self.unhidden {
            output.push(self.nodes.get(&idx).unwrap().output);
        }

        output
    }

    fn reset(&mut self){
        // Reset unhidden nodes
        for i in 0..self.unhidden {
            match self.nodes.get_mut(&i) {
                None => panic!("Tried to reset node which does not exist {}",i),
                Some(n) => n.output = 0.0,
            };  
        }
        // Reset hidden nodes
        for i in self.hidden_nodes.iter(){
            match self.nodes.get_mut(i) {
                None => panic!("Tried to reset node which does not exist {}",i),
                Some(n) => n.output = 0.0,
            };
        }
    }
    /// Start with a default topology
    /// Each input connected to every output
    pub fn generate_initial(&mut self){
        // Create input nodes
        for n in 0..self.inputs {
            self.nodes.insert(n, Node::new(0.0, n as f64 * (1.0 / (self.inputs as f64) )));
        }
        for n in self.inputs..self.unhidden {
            self.nodes.insert(n, Node::new(1.0,n as f64 * (1.0 / (self.inputs as f64) )));
        }

        // Add edges
        for i in 0..self.inputs - 1{ // dont want to connect bias to the end
            for j in self.inputs..self.unhidden { 
                self.add_edge(i, j, 1.0);
            }
        }

        self.calc_ordered_nodes();
    }
}

pub fn create_child(p1: &Genome ,p2: &Genome) -> Genome{
    // Find the more performant parent
    let performer: &Genome;
    let underperformer: &Genome;
    let mut rng = rand::thread_rng();
    if p1.fitness > p2.fitness {
        performer = p1;
        underperformer = p2;
    }else {
        performer = p2;
        underperformer = p1;
    }
    let mut child = Genome::new(performer.inputs, performer.outputs);
    // As we keep all the info from the performer and the mix on the one which are shared
    child = performer.clone();
    for ij in performer.edge_list.iter() {
        if underperformer.edges.contains_key(ij) {
            // we want to copy this edge info over that of the performer
            let p = rng.gen_range(0.0..1.0);
            if p > PROB_CHOOSE_PERFORMER {
                // Take gene from the underperforming parent
                let e = child.edges.get_mut(ij).unwrap();
                let ep = underperformer.edges.get(ij).unwrap();

                e.weight = ep.weight;
                e.enabled = ep.enabled;
            }

        }
    }

    // inheriting node biases
    // Handle output nodes
    for i in performer.inputs..performer.outputs {
        let p = rng.gen_range(0.0..1.0);
        if p > PROB_CHOOSE_PERFORMER {
            let n = child.nodes.get_mut(&i).unwrap();
            let np = underperformer.nodes.get(&i).unwrap();
            n.bias = np.bias;
        }
    }

    for i in performer.hidden_nodes.iter() {
        let p = rng.gen_range(0.0..1.0);
        // If the node is not found skip over it
        if !underperformer.nodes.contains_key(i) {
            continue;
        }
        // Otherwise see if we should swap them
        if p > PROB_CHOOSE_PERFORMER {
            let n = child.nodes.get_mut(i).unwrap();
            let np = underperformer.nodes.get(i).unwrap();
            n.bias = np.bias;
        }
    }

    child.fitness = 0.0; // Reset the fitness
    child
}

pub fn create_mutant(p1: &Genome, popInfo:&mut PopulationInfo) -> Genome{
    let mut child = p1.clone();
    let mut rng = rand::thread_rng();
    // ADD EDGE
    if rng.gen_range(0.0..1.0) < PROB_ADD_EDGE {
        // Add a new edge
        // I cannot be an output and J cannot be an input and want i =/ j
        // And also want to feed from smaller x to larger x value - remove potential loops for now!
        let mut i = rng.gen_range(0..p1.hidden + p1.inputs);
        if i >= p1.inputs {
            i = p1.hidden_nodes[( i - p1.inputs) as usize];
        }

        let mut j = rng.gen_range(p1.inputs..p1.unhidden + p1.hidden);
        if j >= p1.unhidden {
            j = p1.hidden_nodes[(j - p1.unhidden) as usize];
        }
        if i != j {
            // go from lower x to higher x
            let xi = child.nodes.get(&i).unwrap().x;
            let xj = child.nodes.get(&j).unwrap().x;
            if xi > xj {
                let t = i;
                i = j;
                j = t;
            }
            if xi != xj {
                child.add_edge(i, j, rng.gen_range(0.0..2.0) - 1.0 );
            }
        }
    }
    
    // ADD NODE
    if rng.gen_range(0.0..1.0) < PROB_ADD_NODE {
        // Choose random edge to split 
        let idx = rng.gen_range(0..child.num_edges);
        let (i,j) = child.edge_list[idx as usize];
        
        let nodeID = match popInfo.new_nodes.get(&(i,j)) {
            // Not found so add to dictionary and get new a new node entry
            None => {
                popInfo.new_nodes.insert((i, j), popInfo.total_genes + 1);
                //println!("New node id {}", popInfo.total_genes + 1);
                popInfo.total_genes += 1;
                popInfo.total_genes
            },
            Some(id) => *id,
        };

        if !child.nodes.contains_key(&nodeID) {
            child.add_node(i, j, nodeID);
        }
    }

    // MUTATE BIAS
    if rng.gen_range(0.0..1.0) < PROB_SHIFT_BIAS {
        child.mutate_random_node_bias();
    }

    // MUTATE BIAS
    if rng.gen_range(0.0..1.0) < PROB_SHIFT_WEIGHT {
        child.mutate_random_edge_weight();
    }

    child
}

fn genome_distance(p1:&Genome, p2:&Genome) -> f64{
    let mut distance = 0.0;
    let mut weight_diff = 0.0;
    let mut disjoint_edges = 0;
    let mut shared_edges = 0.0;
    let n_edges:f64;

    let mut bias_diff = 0.0;
    let mut n_nodes:f64 = 1.0;
    // Get the number of shared and disjoint edges
    for ij in p1.edge_list.iter(){
        let e1 = p1.edges.get(ij).unwrap();

        match p2.edges.get(ij) {
            None => disjoint_edges += 1,
            Some(e2) => {
                weight_diff += (e1.weight - e2.weight).abs();
                shared_edges += 1.0},
        };
    }
    if p2.num_edges > p1.num_edges {
        n_edges = p2.num_edges as f64;
    }else{
        n_edges = p1.num_edges as f64;
    }

    disjoint_edges += (p2.edge_list.len() - shared_edges as usize) as u64;

    // Compute Bias/Node related stats
    for i in p1.hidden_nodes.iter(){
        let n1 = p1.nodes.get(i).unwrap();
        match p2.nodes.get(i) {
            None => continue,
            Some(n2) => {
                bias_diff += (n1.bias - n2.bias).abs();
                n_nodes += 1.0;
            },
        };
    }

    let t1 = EDGE_DISTANCE_WEIGHT * (disjoint_edges as f64) / n_edges;
    let t2 = WEIGHT_DISTANCE_WEIGHT * weight_diff / shared_edges;
    //let t3 = BIAS_DISTANCE_WEIGHT * bias_diff / n_nodes;

    t1 + t2
}

// ======= Speiciation ===========
#[derive(Debug)]
pub struct Specie {
    pub members:Vec<Genome>,
    fitness_history:Vec<f64>,
    fitness_sum:f64,
    max_fitness_history:usize,
}

impl Specie {
    pub fn new(members: Vec<Genome>) -> Self{
        Self {
            members:members,
            fitness_history:Vec::new(),
            fitness_sum:0.0,
            max_fitness_history:20, // number of elements to keep in the fitness history
        }
    }

    pub fn breed(&self, pop_info:&mut PopulationInfo) -> Genome {
        // Choose if we should mutate or not
        let mut rng = rand::thread_rng();
        // Generate new weight
        let p = rng.gen_range(0.0..1.0);
        if p < PROB_MUTATE_INDIVIDUAL || self.members.len() == 1 {
            //Create Mutated Child if we roll mutate or only have 1 member
            // Get random member
            let p = &self.members[rng.gen_range(0..self.members.len())];

            create_mutant(p, pop_info)
        }else {
            // Create a new child from parents
            let (i1, i2) = (rng.gen_range(0..self.members.len()) , rng.gen_range(0..self.members.len()));
            let (p1,p2) = (&self.members[i1], &self.members[i2]);

            create_child(p1, p2)
        }
        
    }

    pub fn cull_genomes(&mut self, fittest_only: bool){
        self.members.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        let mut member_len = self.members.len();
        let target_len = ((member_len as f64) * CULL_PROPORTION ).ceil() as usize;
        while member_len > target_len {
            self.members.pop();
            member_len -= 1
        }
    }

    pub fn update_fitness(&mut self) {
        let total_members = self.members.len() as f64;
        self.fitness_sum = 0.0;
        for g in self.members.iter_mut() {
            g.adjusted_fitness = g.fitness / total_members;
            self.fitness_sum += g.adjusted_fitness;
        }

        self.fitness_history.push(self.fitness_sum);
        if self.members.len()  > self.max_fitness_history {
            self.members.remove(0);
        }

    }

    pub fn get_best(&self) -> Genome{
        let mut best_genome = match self.members.get(0){
            None => panic!("Cannot get best of an empty species"),
            Some(g) => g.clone(),
        };

        for genome in self.members.iter() {
            if genome.fitness > best_genome.fitness{
                best_genome = genome.clone();
            }
        }

        best_genome
    }

    pub fn should_progress(&self) -> bool{
        let mut avg = 0.0;
        let mut n = 0;
        for g in self.members.iter() {
            avg += g.fitness;
            n += 1;
        }
        // if it has not improved since we started tracking the fitness and we have let it live ...
        // the cull it 
        avg > self.fitness_history[0] || n < self.max_fitness_history


    }
}

// POPULATION

pub struct PopulationInfo {
    total_genes: u64,                            // Total number of genes so far in the population
    new_nodes: HashMap<(u64, u64), u64>,          // Map from edges to the node ids when they split so we can make sure they all line up
}


impl PopulationInfo {
    pub fn new(starting_genes:u64)->Self{
        Self {
            total_genes:starting_genes,
            new_nodes:HashMap::new(),
        }
    }
}



pub struct Population {
    pub species:Vec<Specie>,
    pub size:u64,
    generation:u64,
    pub num_species:u64,
    best_performer:Genome,
    pop_info:PopulationInfo,
    // For implementing iterator
    _cur_species:usize,
    _cur_genome:usize,
    // Inputs outputs
    inputs:u64,
    outputs:u64,
}


impl Population {
    pub fn new(inputs: u64, outputs: u64, size: u64) -> Self {
        // create our empty best performer
        let mut g = Genome::new(inputs, outputs);
        g.generate_initial();

        Self {
            species:Vec::new(),
            size:size,
            generation:0,
            num_species:0,
            best_performer:g,
            pop_info:PopulationInfo::new(inputs + outputs),
            _cur_species:0,
            _cur_genome:0,
            inputs:inputs,
            outputs:outputs,
        }
    }

    fn classify_genome(&mut self, g: Genome) {
        // Handle empty population
        if self.num_species == 0 {
            self.species.push(Specie::new(vec![g]));
            self.num_species += 1;
        }
        else {
            // Non empty population :(
            for i in 0..self.species.len() {
                let s = self.species.get(i).unwrap();
                match s.members.get(0) {
                    None => {
                        self.species.remove(i);
                        self.num_species -= 1;
                        continue;
                    },
                    Some(g2) => {
                        if genome_distance(&g, g2) < SPECIES_DELTA {
                            self.species[i].members.push(g);
                            return;
                        }

                    },
                };
            }
            // Belonged to no species so create a new one!
            self.species.push(Specie::new(vec![g]));
            self.num_species += 1;
            //println!("New Species!!");

        }

    }

    // Finds the fittest specie in the population 
    pub fn update_fittest(&mut self){
        // Loop through Each member of each species
        let mut best_performer_ref = &self.species[0].members[0];

        for specie in self.species.iter() {
            for g in specie.members.iter() {
                if g.fitness > best_performer_ref.fitness {
                    best_performer_ref = g;
                }
            }
        }

        self.best_performer = best_performer_ref.clone();
    }
    
    /// Returns a genome of the best performer
    pub fn get_best_performer(&self) -> Genome{
        self.best_performer.clone()
    }

    fn get_population_size(&self) -> u64 {
        let mut output: u64 = 0;
        for specie in self.species.iter() {
            output += specie.members.len() as u64;
        }
        output
    }

    pub fn evolve(&mut self) {
        // Calculate Global fitness sum
        let mut global_fitness_sum = 0.0;

        if self.species.len() > 0 {
            self.update_fittest();
        }

        for specie in self.species.iter_mut() {
            specie.update_fitness();
            global_fitness_sum += specie.fitness_sum;
            for g in specie.members.iter_mut() {
                if g.get_fitness() == 0.0 {
                    println!("specie with fitness 0?");
                }
                g.visited = false; // After evolving want to reset flag saying we have updated them
            }
        }

        // If we have made no progress then mutate all
        if global_fitness_sum == 0.0 {
            println!("No progresss made :( mutating all");
            for specie in self.species.iter_mut() {
                for g in specie.members.iter_mut() {
                    *g = create_mutant(g, &mut self.pop_info);
                }
            }
            return;
        }

        // Otherwise cull species we dont want to survive;
        // putting this in its own block to avoid reusing num_species here
        {
        let num_species = self.species.len();
        for i in 0..num_species {
            if !self.species[num_species - i - 1].should_progress(){
                println!("Found a species which should not progress");
                self.species.remove(num_species - i - 1);
                self.num_species -= 1;
            }
        }
        }
        // Now cull the remaining genomes
        for specie in self.species.iter_mut() {
            specie.cull_genomes(false);
        }

        // Repopulate
        let mut new_genomes:Vec<Genome> = Vec::new();
        // we have to keep track of the size here as we write to a list then add them to the population
        let mut starting_size = self.get_population_size();
        let diff = self.size - starting_size;

        //println!("The size to start with was {}",starting_size);
        for specie in self.species.iter() {
            let ratio = specie.fitness_sum / global_fitness_sum;
            //println!("Catching overflow magic {} {}",self.size, starting_size);

            let  mut offspring = (ratio * (diff as f64)).ceil() as u64;
            offspring = (self.size - starting_size).min(offspring);

            //println!("this specie needs offspring {}", offspring);
            for _j in 0..offspring { 
                let g = specie.breed(&mut (self.pop_info));
                new_genomes.push(g);
                starting_size += 1;
            }
        }

        // awkward work around for current problem of having a reference to specie inside of self and a reference to self...
        for idx in 0..new_genomes.len() {
            self.classify_genome(new_genomes[idx].clone());
        }

        if self.species.len() == 0 {
            println!("Starting new population");
            for i in 0..self.size {
                let mut g:Genome;
                if i%3 == 0{
                    g = create_mutant(&self.best_performer, &mut self.pop_info);
                }   
                else{
                    g = Genome::new(self.inputs, self.outputs);
                    g.generate_initial();
                    g = create_mutant(&g, &mut self.pop_info);
                }
            self.classify_genome(g)
            }
        }

        self.generation += 1;

    }

    pub fn populate(&mut self) {
        for _i in 0..self.size {
            let mut g:Genome;
            g = Genome::new(self.inputs, self.outputs);
            g.generate_initial();
            g = create_mutant(&g, &mut self.pop_info);   
            self.classify_genome(g)
        }
    }
}
