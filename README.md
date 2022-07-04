<div align="center">

  <h1>Motivus algorithm using <code>wasm-pack</code></h1>

  <strong>A template for kick starting a Rust and WebAssembly project powered by <a href="https://motivus.cl/">Motivus Waterbear Cluster</a> processing  power.</strong>

  <h3>
    <a href="https://motivus.cl/documentation/">Learn more</a>
    <span> | </span>
    <a href="https://discord.gg/t8f5xNhTJW">Join us on Discord</a>
  </h3>
</div>

## 🚴 Usage

### 🐑 Use `cargo generate` to Clone this Template

[Learn more about `cargo generate` here.](https://github.com/ashleygwilliams/cargo-generate)

```
cargo generate --git https://github.com/m0tivus/wasm-pack-template.git --name my-project
cd my-project
```

## 🛠️ Development Environment Requirements
* Docker
   * Your user must belong to the `docker` group.
* Python = 3.7 | 3.8 | 3.9
   * We recommend using a `conda` environment.
* [*Motivus CLI tool* and *Motivus Client library*](https://pypi.org/project/motivus/): `$ pip install motivus`

### 🛠️ Build with `motivus build`

```
$ motivus build
```

### 🔬 Test locally with `motivus loopback`

```
$ motivus loopback
```
```
$ WEBSOCKET_URI=ws://localhost:7070/client_socket/websocket python driver.py
```

### 🎁 Publish to Motivus Marketplace with `motivus push`

```
$ motivus push
```

## 🔋 Batteries Included

* [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen) for communicating
  between WebAssembly and JavaScript.
* [`console_error_panic_hook`](https://github.com/rustwasm/console_error_panic_hook)
  for logging panic messages to the developer console.
* [`wee_alloc`](https://github.com/rustwasm/wee_alloc), an allocator optimized
  for small code size.
