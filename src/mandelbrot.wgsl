@group(0) @binding(0) var<storage, read_write> mandelbrot: array<u32>;
const ITERATIONS: u32 = 100;
// image size is 3500x2000
// workgroup size 10x10
// dispatch 350x200 workgroups
@compute
@workgroup_size(10, 10)
fn main(
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
        ) {
  let x = workgroup_id.x * 10 + local_invocation_id.x;
  let y = workgroup_id.y * 10 + local_invocation_id.y;
  let global_id = (y * 3500 + x) * 3;

  let cx = f32(x) / 1000.0 - 2.5;
  let cy = f32(y) / 1000.0 - 1.0;

  var zx = 0.0;
  var zy = 0.0;
  var zx_square = 0.0;
  var zy_square = 0.0;
  var i = u32(0);

  while(i < ITERATIONS && zx_square + zy_square < 4.0) {
    let prev_zx = zx;
    zx = zx_square - zy_square + cx;
    zy = 2.0 * prev_zx * zy + cy;
    i = i + 1;
    zx_square = zx * zx;
    zy_square = zy * zy;
  }
  if (i == ITERATIONS) {
    mandelbrot[global_id] = 255;
    mandelbrot[global_id + 1] = 255;
    mandelbrot[global_id + 2] = 255;
  }
}
