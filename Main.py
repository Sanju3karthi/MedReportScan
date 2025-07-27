print("Main.py started")
from Utils.Agent import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

with open("Medical_Report.txt", "r") as file:
    medical_report = file.read()

agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

def get_response(agent_name, agent):
    print(f"→ Running agent: {agent_name}")
    response = agent.run()
    print(f"→ {agent_name} response received: {str(response)[:200]}...\n")
    return agent_name, response

responses = {}

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}

    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response

print("\n✅ All individual responses collected.\n")

# Check if all responses exist
for name, resp in responses.items():
    print(f"{name} Response: {str(resp)[:300]}\n")

team_agents = MultidisciplinaryTeam(
    cardiologist_report=responses["Cardiologist"],
    psychologist_report=responses["Psychologist"],
    pulmonologist_report=responses["Pulmonologist"]
)

print("🏥 Running Multidisciplinary Team analysis...")
final_diagnosis = team_agents.run()
print("📄 Final Diagnosis:\n", final_diagnosis)

txt_output_path = "results/final_diagnosis.txt"
os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

with open(txt_output_path, "w") as txt_file:
    txt_file.write(str(final_diagnosis))

print(f"\n✅ Final Diagnosis has been saved to {txt_output_path}")
