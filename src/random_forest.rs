mod dtree;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub fn bootstrap_random_forest(x: &Vec<Vec<f32>>, y: &Vec<String>, n_features: usize, rng: &mut rand::rngs::StdRng) -> (Vec<Vec<f32>>, Vec<String>) { 
    
    let mut x_: Vec<Vec<f32>> = vec![];
    let mut y_: Vec<String> = vec![];

    
    // Creating features indexes
    let mut ft_ixs: Vec<usize> = vec![];
    let mut rnd_ft_idx: usize;
    for _i in 0..n_features {
        rnd_ft_idx = rng.gen_range(0..x[0].len());
        while ft_ixs.contains(&rnd_ft_idx) {
            rnd_ft_idx = rng.gen_range(0..x[0].len());
            
        }
        
        ft_ixs.push(rnd_ft_idx);
    }
    //println!("{:?}", ft_ixs);
    let n_samples = x.clone().len();
    for _i in 0..n_samples {
        let rnd_idx = rng.gen_range(0..x.len());

        let mut x_i: Vec<f32> = vec![];
        for j in 0..n_features {
            x_i.push(x[rnd_idx][j].clone());
        }
        x_.push(x_i.clone());
        y_.push(y[rnd_idx].clone());
    }

    return (x_, y_)
}

pub fn swap_matrix_axes(matrix: &Vec<Vec<String>>) -> Vec<Vec<String>>{
    let mut new_matrix: Vec<Vec<String>> = vec![];
    for j in 0..matrix[0].len(){
        let mut aux: Vec<String> = vec![];
        for i in 0..matrix.len() {
            let s_: String = matrix[i][j].clone();
            
            aux.push(s_);
        }
        new_matrix.push(aux);
    }

    new_matrix

}

pub fn count_val(el: String, y: &Vec<String>) -> usize{
    let mut c: usize = 0;
    for i in 0..y.len() {
        if y[i] == el {
            c = c + 1;
        }
    }c
}

pub fn get_most_common(y: &Vec<String>) -> String {
    let mut common_count: usize = 0;
    let mut common_elem: String = "".to_string();
    for i in 0..y.len() {
        let count = count_val(y[i].clone(), &y);
        if count > common_count {
            common_count = count;
            common_elem = y[i].clone();
        }
    }
    common_elem

}

pub fn accuracy(y: &Vec<String>, y_hat: &Vec<String>) -> f32{
    let mut sum = 0.0;
    for i in 0..y.len(){
        if y[i] == y_hat[i] {
            sum += 1.0;
        }
    }
    sum / (y.len() as f32)
    
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RandomForest {
    n_trees: usize,
    min_samples_split: usize,
    max_depth: usize,
    n_feats: usize,
    seed: u64,
    trees: Vec<dtree::DecisionTreeClassifier>
}

impl RandomForest {
    pub fn new(n_trees: usize, min_samples_split: usize, max_depth: usize, n_feats: usize, seed: u64) -> Self {
        let t: Vec<dtree::DecisionTreeClassifier> = Vec::new();
        RandomForest {
            n_trees: n_trees,
            min_samples_split: min_samples_split,
            max_depth: max_depth,
            n_feats: n_feats,
            seed: seed,
            trees: t
        }
    }

    pub fn fit(&mut self, x: &Vec<Vec<f32>>, y: &Vec<String>) {
        
        let t: Vec<dtree::DecisionTreeClassifier> = Vec::new();
        self.trees = t;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        for i in 0..self.n_trees{
            let mut tree: dtree::DecisionTreeClassifier = dtree::DecisionTreeClassifier::new(self.min_samples_split, self.max_depth);
            let (x_sample, y_sample) = bootstrap_random_forest(&x, &y, self.n_feats, &mut rng);
            
            //let (x_sample, y_sample) = bootstrap_random_forest(&x, &y, 2, self.n_feats, &mut rng);
            tree.fit(&x_sample, &y_sample);
            self.trees.push(tree);
            //println!("{:?}", x_sample);
            println!("{:?}", i);
        }

        
    }

    pub fn predict(&mut self, x: &Vec<Vec<f32>>) -> Vec<String> {
        let mut tree_preds: Vec<Vec<String>> = vec![];
        for i in 0..self.n_trees {
            let y_pred_i = self.trees[i].predict(&x);
            tree_preds.push(y_pred_i);
        }
        //println!("{}", tree_preds.len());
        let tree_preds_ = swap_matrix_axes(&tree_preds);
        //println!("{}", tree_preds_.len());
        let mut y_pred = vec![];
        
        for i in 0..tree_preds_.len() {
            let most_common_label = get_most_common(&tree_preds_[i]);
            y_pred.push(most_common_label)
        }
        y_pred

    }

}