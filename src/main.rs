use macroquad::prelude::*;
pub mod neat;
use neat::{Genome, Population};
use ::rand::prelude::*;


fn draw_genome(g: &Genome,x: f32,y: f32, w: f32, h: f32) {
    // Draw in and out nodes
    for i in 0..g.inputs + g.outputs {
        let n = &g.nodes[&i];
        let x1 = x + w*n.x as f32;
        let y1 = y + h * n.y as f32;
        draw_circle(x1 ,y1, 5.0, GREEN);

        // Draw connections
        for j in n.from.iter() {
            let n2 = &g.nodes[j];
            let x2 = x + w*n2.x as f32;
            let y2 = y + h * n2.y as f32;
            draw_line(x1, y1, x2, y2, 5.0, BLUE);

        }
    }

    for i in g.hidden_nodes.iter() {
        let n = &g.nodes[i];
        let x1 = x + w*n.x as f32;
        let y1 = y + h * n.y as f32;
        draw_circle(x1 ,y1, 5.0, GREEN);

        // Draw connections
        for j in n.from.iter() {
            let n2 = &g.nodes[j];
            let x2 = x + w*n2.x as f32;
            let y2 = y + h * n2.y as f32;
            draw_line(x1, y1, x2, y2, 5.0, BLUE);

        }
    }


}

/// Draws the bar graph with population sizes
fn draw_pop_sizes(p: &Population,x: f32,y: f32,w: f32,h: f32){
    let len = p.species.len() as f32;
    for i in 0..p.species.len(){
        let t:f32 = w * (i as f32) / len;
        draw_line(x + t, y + h, x + t , y + h -  h * (p.species[i].members.len() as f32 / p.size as f32), 2.0, RED);
    }

}


/// Draws the bar graph with population sizes
fn draw_pop_fitnesses(p: &Population,x: f32,y: f32,w: f32,h: f32){
    let len = p.species.len() as f32;
    let best_fitness = p.get_best_performer().get_fitness() as f32;
    for i in 0..p.species.len(){
        let t:f32 = w * (i as f32) / len;
        if p.species[i].get_best().get_fitness() as f32 > best_fitness {
            println!("Umm fitness is > our best fitness ???");
        }  
        let temp = p.species[i].get_best().get_fitness() as f32;
        let scoring = (temp as f32 - best_fitness ).abs() / best_fitness;
        if temp > best_fitness {
            println!("temp was greater than best fitness ummm");
        }
        draw_line(x + t, y + h, x + t , y + h -  h * (1.0 - scoring), 2.0, GREEN);

    }
}

#[macroquad::main("MyGame")]
async fn main() {
    let mut p = Population::new(3, 1, 500);
    p.populate();
    let mut rng = ::rand::thread_rng();
    let mut gen = 0;

    let g = p.get_best_performer();
    draw_genome(&g, 30.0, 0.0, 500.0, 100.0);   
    draw_pop_sizes(&p,540.0,0.0,200.0,100.0);     
    draw_pop_fitnesses(&p,540.0,120.0,200.0,100.0);  

    loop {

        for _j in 0..50 {
            // Loop through members
            for s in p.species.iter_mut(){
                for g in s.members.iter_mut() {
                    if g.visited == true {
                        continue;
                    }
                    let mut u = 0.0;
                    let mut b = 0.0;
                    b = g.forward(vec![1.0, 1.0, 1.0])[0].powi(2);
                    u+=b;
                    b = g.forward(vec![0.0, 0.0, 1.0])[0].powi(2);
                    u+=b;
                    b = (1.0 - g.forward(vec![1.0, 0.0, 1.0])[0]).powi(2);
                    u+=b;
                    b = (1.0 - g.forward(vec![0.0, 1.0, 1.0])[0]).powi(2);
                    u+=b;
                    u = u * 1000.0;
                    g.set_fitness(1.0/u);
                    g.visited = true;
                }
            }
            //println!("Evolving!!");
            p.evolve();
        }
        let mut g = p.get_best_performer();
        draw_genome(&g, 30.0, 0.0, 500.0, 100.0);   
        draw_pop_sizes(&p,540.0,0.0,200.0,100.0);     
        draw_pop_fitnesses(&p,540.0,120.0,200.0,100.0);     

        //draw some of the top 10 genomes!!!
        for i in 0..p.num_species.min(20) {
            draw_genome(&p.species[i as usize].get_best() , 30.0 + ((i % 5) *100 ) as f32, 150.0 + (70 * (i/5)) as f32, 90.0, 60.0);   
            
        }

        if gen % 500 == 0 {
            println!("===Generation {}===",gen);
            println!("Species: {}",p.species.len());
            println!("Fitness:{}",g.get_fitness());
            println!("0 xor 0 {}", g.forward(vec![0.0, 0.0, 1.0])[0].abs());
            println!("1 xor 1 {}", g.forward(vec![1.0, 1.0, 1.0])[0].abs());
            println!("1 xor 0 {}", g.forward(vec![1.0, 0.0, 1.0])[0].abs());
            println!("0 xor 1 {}", g.forward(vec![0.0, 1.0, 1.0])[0].abs());

        }
        gen += 50;
        // Update pipes
        // Run agents
        // Do collisions
        // Draw
        // Go next frame
        next_frame().await
    }


}