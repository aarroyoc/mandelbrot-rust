use image::RgbImage;
use std::arch::x86_64::{_mm256_add_pd, _mm256_cmp_pd, _mm256_div_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_set_pd, _mm256_sub_pd, _mm256_testz_pd};
use std::path::Path;
use std::time::Instant;
use std::sync::{Mutex, Arc};
use rayon::prelude::*;

const ITERATIONS: u32 = 100;
const WIDTH: u32 = 3500;
const HEIGHT: u32 = 2000;
const BUFFER_SIZE: usize = (WIDTH * HEIGHT * 3) as usize;


struct Complex {
    real: f64,
    im: f64,
}

impl Complex {
    fn square(&mut self) {
	let prev_real = self.real;
	self.real = self.real * self.real - self.im * self.im;
	self.im = 2.0 * prev_real * self.im;
    }

    fn plus(&mut self, other: &Complex) {
	self.real = self.real + other.real;
	self.im = self.im + other.im;
    }

    fn abs(&self) -> f64 {
	(self.real * self.real + self.im * self.im).sqrt()
    }
}

fn main() {
    // sequential
    println!("Starting Sequential algorithm");
    let mut buffer: Vec<u8> = vec![0; BUFFER_SIZE];
    let now = Instant::now();
    for x in 0..WIDTH {
	for y in 0..HEIGHT {
	    let c = Complex {
		real: x as f64 / 1000.0 - 2.5,
		im: y as f64 / 1000.0 - 1.0,
	    };
	    let mut z = Complex {
		real: 0.0,
		im: 0.0,
	    };
	    let mut i = 0;
	    
	    while i < ITERATIONS && z.abs() < 2.0 {
		z.square();
		z.plus(&c);
		i = i + 1;
	    }
	    if i == ITERATIONS {
		let index = ((y * WIDTH + x) * 3) as usize;
		buffer[index] = 255;
		buffer[index+1] = 255;
		buffer[index+2] = 255;
	    }
	}
    }
    let elapsed = now.elapsed();
    println!("Time for Sequential algorithm: {}ms", elapsed.as_millis());    
    save_fractal(buffer, &Path::new("sequential.png"));

    // optimized sequential
    println!("Starting Optimized Sequential algorithm");
    let mut buffer: Vec<u8> = vec![0; BUFFER_SIZE];
    let now = Instant::now();
    for x in 0..WIDTH {
	for y in 0..HEIGHT {
	    let cx = x as f64 / 1000.0 - 2.5;
	    let cy = y as f64 / 1000.0 - 1.0;
	    let mut zx = 0.0;
	    let mut zy = 0.0;
	    let mut zx_square = 0.0;
	    let mut zy_square = 0.0;
	    let mut i = 0;
	    
	    while i < ITERATIONS && zx_square + zy_square < 4.0 {
		let prev_zx = zx;
		zx = zx_square - zy_square + cx;
		zy = 2.0 * prev_zx * zy + cy;
		i = i + 1;
		zx_square = zx * zx;
		zy_square = zy * zy;
	    }
	    if i == ITERATIONS {
		let index = ((y * WIDTH + x) * 3) as usize;
		buffer[index] = 255;
		buffer[index+1] = 255;
		buffer[index+2] = 255;
	    }
	}
    }
    let elapsed = now.elapsed();
    println!("Time for Optimized Sequential algorithm: {}ms", elapsed.as_millis());    
    save_fractal(buffer, &Path::new("opti-sequential.png"));

    // SIMD
    if is_x86_feature_detected!("avx2") {
	println!("AVX2 detected");
	println!("Starting SIMD AVX2 algorithm");
	let mut buffer: Vec<u8> = vec![0; BUFFER_SIZE];
	let now = Instant::now();
	unsafe {
	    let zeros = _mm256_set1_pd(0.0);
	    let two = _mm256_set1_pd(2.0);
	    let four = _mm256_set1_pd(4.0);
	    let ones = _mm256_set1_pd(1.0);
	    for x in 0..WIDTH {
		let mut y = 0;
		while y < HEIGHT {
		    let mut xs = _mm256_set1_pd(x as f64);
		    let mut ys = _mm256_set1_pd(y as f64);
		    let mut a = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);
		    ys = _mm256_add_pd(ys, a);
		    a = _mm256_set1_pd(1000.0);
		    xs = _mm256_div_pd(xs, a);
		    ys = _mm256_div_pd(ys, a);
		    a = _mm256_set1_pd(2.5);
		    let cxs = _mm256_sub_pd(xs, a);
		    let cys = _mm256_sub_pd(ys, ones);
		    let mut zxs = zeros;
		    let mut zys = zeros;
		    let mut cmp = ones;
		    for _ in 0..ITERATIONS {
			let zx_square = _mm256_mul_pd(zxs, zxs);
			let zy_square = _mm256_mul_pd(zys, zys);
			let abs = _mm256_add_pd(zx_square, zy_square);
			cmp = _mm256_cmp_pd(abs, four, 1);
			if _mm256_testz_pd(cmp, cmp) == 1 {
			    break;
			}
			
			let previous_zxs = zxs;
			zxs = _mm256_sub_pd(zx_square, zy_square);
			zxs = _mm256_add_pd(zxs, cxs);
			zys = _mm256_mul_pd(two, zys);
			zys = _mm256_mul_pd(zys, previous_zxs);
			zys = _mm256_sub_pd(zys, cys);
		    }

		    let (cmp3, cmp2, cmp1, cmp0): (f64, f64, f64, f64) = std::mem::transmute(cmp);
		    if cmp3 != 0.0 {
			let index = (((y + 3) * WIDTH + x) * 3) as usize;
			buffer[index] = 255;
			buffer[index+1] = 255;
			buffer[index+2] = 255;
		    }

		    if cmp2 != 0.0 {
			let index = (((y + 2) * WIDTH + x) * 3) as usize;
			buffer[index] = 255;
			buffer[index+1] = 255;
			buffer[index+2] = 255;
		    }

		    if cmp1 != 0.0 {
			let index = (((y + 1) * WIDTH + x) * 3) as usize;
			buffer[index] = 255;
			buffer[index+1] = 255;
			buffer[index+2] = 255;
		    }

		    if cmp0 != 0.0 {
			let index = ((y * WIDTH + x) * 3) as usize;
			buffer[index] = 255;
			buffer[index+1] = 255;
			buffer[index+2] = 255;
		    }
		    y = y + 4;
		}
	    }
	}
	let elapsed = now.elapsed();
	println!("Time for SIMD AVX2 algorithm: {}ms", elapsed.as_millis());    
	save_fractal(buffer, &Path::new("simd-avx2-sequential.png"));
    }

    // threads
    println!("Starting Threads algorithm");
    let buffer: Vec<u8> = vec![0; BUFFER_SIZE];
    let buffer = Arc::new(Mutex::new(buffer));
    let now = Instant::now();
    let cores: usize = std::thread::available_parallelism().unwrap().into();
    let cores: u32 = cores as u32;
    println!("Using {} cores", cores);
    let mut threads = vec![];
    for core in 0..cores {
	let x_start = core*WIDTH/cores;
	let x_end = if core == cores-1 {
	    WIDTH
	} else {
	    (core+1)*WIDTH/cores
	};
	let buffer_rc = Arc::clone(&buffer);
	threads.push(std::thread::spawn(move || {
	    for x in x_start..x_end {
		for y in 0..HEIGHT {
		    let c = Complex {
			real: x as f64 / 1000.0 - 2.5,
			im: y as f64 / 1000.0 - 1.0,
		    };
		    let mut z = Complex {
			real: 0.0,
			im: 0.0,
		    };
		    let mut i = 0;
		    while i < ITERATIONS && z.abs() < 2.0 {
			z.square();
			z.plus(&c);
			i = i + 1;
		    }
		    if i == ITERATIONS {
			let mut buffer = buffer_rc.lock().unwrap();
			let index = ((y * WIDTH + x) * 3) as usize;
			buffer[index] = 255;
			buffer[index+1] = 255;
			buffer[index+2] = 255;
		    }
		}
	    }
	}));
    }
    for t in threads {
	t.join().unwrap();
    }
    let elapsed = now.elapsed();
    println!("Time for threaded algorithm: {}ms", elapsed.as_millis());
    let buffer: Vec<u8> = std::mem::take(&mut buffer.lock().unwrap());
    save_fractal(buffer, &Path::new("threaded.png"));

    // rayon
    println!("Starting rayon algorithm");
    let now = Instant::now();
    let buffer: Vec<u8> = (0..HEIGHT).into_par_iter().flat_map_iter(|y| {
	let mut column = Vec::with_capacity(WIDTH as usize * 3);
	for x in 0..WIDTH {
	    let c = Complex {
		real: x as f64 / 1000.0 - 2.5,
		im: y as f64 / 1000.0 - 1.0,
	    };
	    let mut z = Complex {
		real: 0.0,
		im: 0.0,
	    };
	    let mut i = 0;
	    while i < ITERATIONS && z.abs() < 2.0 {
		z.square();
		z.plus(&c);
		i = i + 1;
	    }
	    if i == ITERATIONS {
		column.extend_from_slice(&[255, 255, 255]);
	    } else {
		column.extend_from_slice(&[0, 0, 0]);
	    }
	}
	column
    })
	.collect();
		
    let elapsed = now.elapsed();
    println!("Time for rayon algorithm: {}ms", elapsed.as_millis());
    save_fractal(buffer, &Path::new("rayon.png"));

    // rayon pre-allocated
    println!("Starting rayon pre-allocated algorithm");
    let now = Instant::now();
    let mut buffer: Vec<u8> = vec![0; BUFFER_SIZE];

    buffer
	.par_chunks_exact_mut(3)
	.enumerate()	
	.for_each(|(idx, chunk)| {
	    let x = (idx as u32) % WIDTH;
	    let y = (idx as u32) / WIDTH;

	    let c = Complex {
		real: x as f64 / 1000.0 - 2.5,
		im: y as f64 / 1000.0 - 1.0,
	    };
	    let mut z = Complex {
		real: 0.0,
		im: 0.0,
	    };
	    let mut i = 0;
	    while i < ITERATIONS && z.abs() < 2.0 {
		z.square();
		z.plus(&c);
		i = i + 1;
	    }
	    if i == ITERATIONS {
		chunk[0] = 255;
		chunk[1] = 255;
		chunk[2] = 255;
	    }
	});
    
    let elapsed = now.elapsed();
    println!("Time for rayon-preallocated algorithm: {}ms", elapsed.as_millis());
    save_fractal(buffer, &Path::new("rayon-preallocated.png"));    
}


fn save_fractal(buffer: Vec<u8>, path: &Path) {
    let img = RgbImage::from_raw(WIDTH, HEIGHT, buffer).unwrap();

    img.save(path).unwrap();
}
