use nannou::{
    prelude::*,
    rand::prelude::*,
    glam::Vec3Swizzles,
    color::Gradient,
};
use std::collections::VecDeque;
use rhai::{Engine, AST, Scope, OptimizationLevel};
use nannou_egui::{self, egui, Egui};

const INITIAL_BOX: f32 = 100.0;
const PARTICLES: usize = 250;
const HISTORY_POINTS: usize = 100;

struct Particle {
    pos: Vec3,
    history: VecDeque<Vec3>,
}

impl Particle {
    fn new(pos: Vec3) -> Self {
        Self {
            pos,
            history: VecDeque::with_capacity(HISTORY_POINTS),
        }
    }

    fn update_position(&mut self, new: Vec3) {
        while self.history.len() >= HISTORY_POINTS {
            self.history.pop_back();
        }
        self.history.push_front(self.pos);
        self.pos = new;
    }
}

#[derive(Debug)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn new(point: Vec3) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    fn ease_to(&mut self, other: Bounds, factor: f32) {
        self.min.x += factor * (-self.min.x + other.min.x);
        self.min.y += factor * (-self.min.y + other.min.y);
        self.min.z += factor * (-self.min.z + other.min.z);
        self.max.x += factor * (-self.max.x + other.max.x);
        self.max.y += factor * (-self.max.y + other.max.y);
        self.max.z += factor * (-self.max.z + other.max.z);
    }

    fn add_point(&mut self, point: Vec3) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }

    fn whd(&self) -> Vec3 {
        self.max - self.min
    }

    fn max_axis(&self) -> f32 {
        let whd = self.whd();
        whd.x.max(whd.y).max(whd.z)
    }

    fn xyz(&self) -> Vec3 {
        (self.max + self.min) / 2.0
    }
}

struct Model {
    bounds: Bounds,
    particles: Vec<Particle>,
    engine: Engine,
    script: AST,
    egui: Egui,
}

impl Model {
    fn new(app: &App) -> Self {
        let mut rng = thread_rng();
        let mut particles = (0..PARTICLES)
            .map(|_| Particle::new(Vec3::new(
                rng.gen_range(-INITIAL_BOX..=INITIAL_BOX),
                rng.gen_range(-INITIAL_BOX..=INITIAL_BOX),
                rng.gen_range(-INITIAL_BOX..=INITIAL_BOX),
            )))
            .collect();

        let window_id = app
            .new_window()
            .view(view)
            .raw_event(raw_window_event)
            .build()
            .unwrap();
        let window = app.window(window_id).unwrap();
        let egui = Egui::from_window(&window);

        let mut engine = Engine::new();
        engine.set_optimization_level(OptimizationLevel::Full);
        engine
            .register_type_with_name::<Vec3>("Vec3")
            .register_fn("vec3", Vec3::new)
            .register_fn("*", |a: Vec3, b: f32| a * b)
            .register_fn("*", |a: f32, b: Vec3| a * b)
            .register_fn("/", |a: Vec3, b: f32| a / b)
            .register_fn("/", |a: f32, b: Vec3| a / b)
            .register_fn("+", |a: Vec3, b: Vec3| a + b)
            .register_fn("-", |a: Vec3, b: Vec3| a - b)
            .register_get_set("x", |v: &mut Vec3| v.x, |v: &mut Vec3, a: f32| v.x = a )
            .register_get_set("y", |v: &mut Vec3| v.y, |v: &mut Vec3, a: f32| v.y = a )
            .register_get_set("z", |v: &mut Vec3| v.z, |v: &mut Vec3, a: f32| v.z = a );
        engine
            .register_fn("sin", f32::sin)
            .register_fn("cos", f32::sin)
            .register_fn("tan", f32::sin);
        let script = engine.compile(include_str!("script.rhai")).unwrap();

        Self {
            bounds: Bounds::new(Vec3::ZERO),
            particles,
            engine,
            script, 
            egui,
        }
    }
}

fn main() {
    nannou::app(Model::new)
        .update(update)
        .simple_window(view)
        .run();
}

fn update(app: &App, model: &mut Model, update: Update) {
    model.egui.set_elapsed_time(update.since_start);
    let ctx = model.egui.begin_frame();

    let delta = update.since_last.as_secs_f32();
    let mut bounds_target = Bounds::new(model.particles[0].pos);

    for mut particle in model.particles.iter_mut() {
        let mut pos = particle.pos;
        for _ in 0..10 {
            pos = model.engine.call_fn::<Vec3>(
                &mut Scope::new(),
                &model.script,
                "tick",
                (pos, delta / 100.0)
            ).unwrap();
        }
        particle.update_position(pos);
        bounds_target.add_point(particle.pos);
    }

    if model.bounds.max_axis() > bounds_target.max_axis() && model.bounds.max_axis() < 1.5 * bounds_target.max_axis() {
    } else {
        model.bounds.ease_to(bounds_target, delta);
    }

    egui::Window::new("Bounds").show(&ctx, |ui| {
        ui.label(format!("{:#?}", model.bounds));
    });
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    // Let egui handle things like keyboard and mouse input.
    model.egui.handle_raw_event(event);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app
        .draw();

    draw.background().color(BLACK);
    let center = model.bounds.xyz();
    let biggest_side = model.bounds.max_axis();
    let window_rect = app.window_rect();
    let min_dim = window_rect.w().min(window_rect.h());
    let mat =
        Mat4::from_translation(-center)
        * Mat4::from_rotation_y(app.time / 10.0)
        * Mat4::orthographic_lh(
            biggest_side,
            -biggest_side,
            biggest_side,
            -biggest_side,
            biggest_side,
            -biggest_side,
        )
        * Mat4::from_scale(Vec3::ONE + Vec3::Z) // ????????
        * Mat4::from_scale(Vec3::ONE * min_dim / 2.0);

    for particle in model.particles.iter() {
        let pos = mat.transform_point3(particle.pos);
        draw.quad()
            .z_degrees(45.0)
            .points(
                Point2::new(1.0, 1.0),
                Point2::new(1.0, -1.0),
                Point2::new(-1.0, -1.0),
                Point2::new(-1.0, 1.0),
            )
            .xyz(pos)
            .color(WHITE);

        let gradient = Gradient::new([WHITE, PINK, LIGHTBLUE, BLACK] 
            .into_iter()
            .map(|c| c.into_format::<f32>())
            .map(|c| c.into_encoding::<nannou::color::encoding::Linear<_>>())
        );
        let mut colors = gradient.take(HISTORY_POINTS + 1);
        if particle.history.len() > 0 {
            draw.polyline()
                .points_colored(Some(pos)
                    .into_iter()
                    .chain(particle
                        .history
                        .iter()
                        .map(|p| mat.transform_point3(*p))
                    )
                    .zip(colors)
                )
                .color(ORANGE);
        }
    }

    //let lines = [
    //    Vec3::X,
    //    Vec3::Y,
    //    Vec3::Z,
    //    -Vec3::X,
    //    -Vec3::Y,
    //    -Vec3::Z,
    //];
    //let start = mat.transform_point3(Vec3::ZERO);
    //for endpoint in lines {
    //    let end = mat.transform_point3(endpoint * biggest_side);
    //    draw.polyline()
    //        .points([start, end])
    //        .color(DARKGREY);
    //}

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

