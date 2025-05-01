import streamlit as st
from streamlit.components.v1 import html

def dot_background():
    html(f"""
    <canvas id="canvasBg"></canvas>
    <style>
        #canvasBg {{
            position: fixed;
            top: 0;
            left: 0;
            z-index: -1;
            width: 100vw;
            height: 100vh;
        }}
        [data-testid="stAppViewContainer"] {{
            background-color: transparent;
        }}
        .stApp {{
            background: linear-gradient(rgba(14, 17, 23, 0.95), rgba(14, 17, 23, 0.95));
        }}
    </style>
    <script>
        const canvas = document.getElementById('canvasBg');
        const ctx = canvas.getContext('2d');
        
        function resizeCanvas() {{
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }}
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        // Particle configuration
        const particles = [];
        const particleCount = {{
            mobile: 30,
            desktop: 80
        }};
        const isMobile = /Mobi|Android/i.test(navigator.userAgent);
        const count = isMobile ? particleCount.mobile : particleCount.desktop;
        
        // Create particles
        class Particle {{
            constructor() {{
                this.reset();
            }}
            reset() {{
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 0.8;
                this.vy = (Math.random() - 0.5) * 0.8;
                this.radius = Math.random() * 2;
                this.color = `hsl(${{Math.random() * 360}}, 70%, 60%)`;
            }}
            update() {{
                this.x += this.vx;
                this.y += this.vy;
                
                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
            }}
        }}

        // Initialize particles
        for (let i = 0; i < count; i++) {{
            particles.push(new Particle());
        }}

        function animate() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Update and draw particles
            particles.forEach(particle => {{
                particle.update();
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
                ctx.fillStyle = particle.color;
                ctx.fill();
            }});

            // Draw connections
            particles.forEach((a, i) => {{
                particles.slice(i + 1).forEach(b => {{
                    const dx = a.x - b.x;
                    const dy = a.y - b.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < 150) {{
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y);
                        ctx.lineTo(b.x, b.y);
                        ctx.strokeStyle = a.color;
                        ctx.globalAlpha = 1 - dist/150;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }}
                }});
            }});
            
            ctx.globalAlpha = 1;
            requestAnimationFrame(animate);
        }}
        
        animate();
    </script>
    """, height=0, width=0)
