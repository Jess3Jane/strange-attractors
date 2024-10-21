fn tick(pos, delta) {
	const sigma = 10;
	const rho = 28;
	const beta = 8 / 3;

	let dx = sigma * (-pos.x + pos.y);
	let dy = -pos.x * pos.z + rho * pos.x - pos.y;
	let dz = pos.x * pos.y - beta * pos.z;
	
	pos + vec3(dx, dy, dz) * delta
}
