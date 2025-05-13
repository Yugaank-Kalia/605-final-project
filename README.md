# Music Recommender Setup Guide

This guide walks you through setting up the music recommender system using AWS EC2, Spotify API, OpenAI API, Next,js and ngrok.

---

## Environment Setup

Before proceeding, set up a `.env` file with the following keys:

-   `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`  
    Get from: https://developer.spotify.com/dashboard
-   `OPENAI_API_KEY`  
    Get from: https://platform.openai.com/api-keys
-   `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`  
    Get from: https://aws.amazon.com/console/

---

## Backend Server Setup on AWS EC2

### 1. Launch EC2 Instance

-   Use Ubuntu as the base image.
-   Connect to the instance via SSH.

### 2. Prepare Directory

```bash
mkdir recommender
```

### 3. Transfer Files to EC2

From your local machine with AWS key pair and backend files:

```bash
scp -i "path/to/key.pem" path/to/server.py ubuntu@<EC2-IP>:/home/ubuntu/recommender/server.py
scp -i "path/to/key.pem" path/to/.env ubuntu@<EC2-IP>:/home/ubuntu/recommender/.env
scp -i "path/to/key.pem" path/to/requirements.txt ubuntu@<EC2-IP>:/home/ubuntu/recommender/requirements.txt
```

### 4. SSH Into EC2

```bash
ssh -i "path/to/key.pem" ubuntu@<EC2-IP>
```

### 5. Set Up Python Environment

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 6. Launch the Flask Server

```bash
python server.py
```

---

## Ngrok Tunnel for React Frontend

In a separate terminal window, SSH into the EC2 again and set up ngrok:

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok
```

Authenticate ngrok:

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

Start the tunnel:

```bash
ngrok http 5000
```

**Copy the forwarding address** and update the `page.tsx` file in the `discover` folder of your React frontend (around line 59).

---

## Test Backend Without Frontend

To test your server locally without the React frontend:

```bash
curl -X POST http://localhost:5000/api/recommend -H "Content-Type: application/json" -d '{"spotify_url": "https://open.spotify.com/track/5CQ30WqJwcep0pYcV4AMNc?si=810aafbc65b1413d"}'
```

---

## You're Ready!

Your Flask backend is now live and connected to Spotify, OpenAI, and optionally, your frontend via ngrok.

---

## Setting Up the Next.js Frontend

Follow these steps to get it running:

### 1. Update API Endpoint

In your frontend code (likely in `frontend/app/discover/page.tsx`):

-   Replace the current API URL with your **ngrok forwarding address** (e.g., `https://abc123.ngrok.io`).

```tsx
const response = await fetch('https://abc123.ngrok.io/api/recommend', {
	method: 'POST',
	headers: {
		'Content-Type': 'application/json',
	},
	body: JSON.stringify({ spotify_url }),
});
```

### 2. Run the Next.js Dev Server

```bash
bun install
bun run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000)

---

## Final Notes

-   Ensure CORS is handled in your Flask server if using ngrok with the frontend.
-   You can add `.env.local` in your frontend with variables like your public API base URL.

---

## Example `.env` File

Create a `.env` file in your backend directory with the following content:

```
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

---

## Infrastructure & Scalability Setup (AWS)

This section outlines how to set up the infrastructure for scaling and deploying the music recommender backend on AWS.

### Notes

-   TO TEST SCALABILITY, use the `no_spotify.py` file. It disables all external Spotify and OpenAI API calls to avoid rate limits.  
    Default setup tests recommendations for "Stairway to Heaven" by Led Zeppelin.  
    To change, update the track and artist name in the `recommend_songs()` function.

-   WHEN DONE TESTING, set your Auto Scaling Group (ASG) desired capacity to 0 and disable automatic scaling to stop all running instances.

---

### AWS S3 Bucket

Upload the following files to your designated S3 bucket:

-   `server.py`
-   `.env`
-   `requirements.txt`
-   `no_spotify.py`

---

### Security Groups

**1. ec2-security-group**

-   Inbound:
    -   Port 5000 from `port80-security-group`
    -   Port 22 from `0.0.0.0/0`
-   Outbound: All traffic

**2. port80-security-group**

-   Inbound: Port 80 from `0.0.0.0/0`
-   Outbound: All traffic

**3. recommender-security-group**

-   Inbound: Ports 80, 443, 5000, 22 from `0.0.0.0/0`
-   Outbound: All traffic

---

### Main EC2 Instance

-   Name: `RecommendationServer`
-   Type: `t2.micro`
-   OS: Ubuntu
-   Key Pair: `MSML605 (1)`
-   Security Group: `ec2-security-group`
-   Availability Zone: `us-east-2a`

---

### Launch Template

-   Name: `recommendation-scaler` (Default version)
-   Instance Type: `t2.micro`
-   AMI: `flask-recommender-ami-v1`
-   Key Pair: `MSML605 (1)`
-   Security Group: `recommender-security-group`
-   Subnet: Do NOT specify (leave default)
-   Advanced Settings:
    -   IAM Instance Profile: `EC2S3AccessRole`
    -   User Data: Paste content from `template_user_data.txt`

---

### AMI

-   Name: `flask-recommender-ami-v1`
-   Built from the running main EC2 instance

---

### Load Balancer

-   Name: `flask alb`
-   Type: Application Load Balancer (ALB)
-   Scheme: Internet-Facing
-   Availability Zones: `us-east-2a`, `us-east-2b`
-   Listeners & Rules:
    -   Port 80: Forward to `recommendation-target-group`
-   Security Group: `port80-security-group`

---

### Target Group

-   Name: `recommendation-target-group`
-   Target Type: Instance
-   Protocol: HTTP
-   Port: 5000
-   Load Balancer: `flask alb`
-   Health Checks:
    -   Protocol: HTTP
    -   Path: `/health`
    -   Port: 5000

---

### Auto Scaling Group (ASG)

-   Name: `flask-asg`
-   Desired Capacity: 1
-   Scaling Limits: Min 1, Max 4
-   Launch Template: `recommendation-scaler` (default version)
-   AMI: `flask-recommender-ami-v1`
-   Instance Type: `t2.micro`
-   Key Pair: `MSML605 (1)`
-   Security Group: `recommender-security-group`
-   Availability Zones: `us-east-2a`, `us-east-2b`
-   Replacement Behavior: Launch before terminating
-   Automatic Scaling:
    -   Type: Target Tracking
    -   Enabled: Yes
    -   Metric: Average CPU utilization at 70%
    -   Scale-In: Enabled

---

## Testing Scalability with Locust

Once your Auto Scaling Group (ASG) is set up in AWS, you can begin testing the server's scalability. You have two options:

-   Test the **baseline EC2 instance** without autoscaling.
-   Test the **autoscaled instance** created via ASG and Load Balancer.

---

### Run Locust for Baseline Instance

```bash
locust --host=http://<ec2-instance-ip>:<PORT NUMBER>
```

---

### Run Locust for Autoscaled Instance (via Load Balancer)

```bash
locust --host=http://<ec2-load-balancer-dns-name>
```

Then open your browser and go to:

```
http://localhost:8089
```

This opens the Locust web interface where you can simulate user traffic.

---

### Monitor Flask Server Logs

To follow server activity on the running EC2 instance:

1. SSH into the instance (or use AWS browser-based connection).
2. Run the following to monitor logs:

```bash
cd ~/recommender
tail -f server.log
```

This shows real-time Flask server activity, including incoming requests triggered by Locust.
