import streamlit.components.v1 as components

def dot_background():
    components.html(
        """
        <canvas id="dotCanvas"></canvas>
        <style>
            #dotCanvas {
                position: fixed;
                top: 0;
                left: 0;
                z-index: 0;
                pointer-events: none;
            }
            .stApp {
                background-color: rgba(14, 17, 23, 0.9) !important;
                position: relative;
                z-index: 1;
            }
        </style>
        <script>
            const canvas = document.createElement('canvas');
            document.body.prepend(canvas);
            const ctx = canvas.getContext('2d');
            
            function resizeCanvas() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }
            resizeCanvas();
            
            const dots = Array(80).fill().map(() => ({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.8,
                vy: (Math.random() - 0.5) * 0.8,
                radius: Math.random() * 2,
                color: ['#9C27B0', '#E91E63', '#2196F3', '#00BCD4'][Math.floor(Math.random() * 4)]
            }));
            
            function animate() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                dots.forEach(dot => {
                    dot.x += dot.vx;
                    dot.y += dot.vy;
                    
                    if (dot.x < 0 || dot.x > canvas.width) dot.vx *= -1;
                    if (dot.y < 0 || dot.y > canvas.height) dot.vy *= -1;
                    
                    ctx.beginPath();
                    ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2);
                    ctx.fillStyle = dot.color;
                    ctx.fill();
                });
                
                dots.forEach((a, i) => {
                    dots.slice(i).forEach(b => {
                        const dx = a.x - b.x;
                        const dy = a.y - b.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        if (distance < 100) {
                            ctx.beginPath();
                            ctx.moveTo(a.x, a.y);
                            ctx.lineTo(b.x, b.y);
                            ctx.strokeStyle = `rgba(156, 39, 176, ${1 - distance/100})`;
                            ctx.lineWidth = 0.3;
                            ctx.stroke();
                        }
                    });
                });
                
                requestAnimationFrame(animate);
            }
            
            window.addEventListener('resize', resizeCanvas);
            animate();
        </script>
        """,
        height=0,
        width=0,
    )
