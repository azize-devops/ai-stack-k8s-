# AI Intern Portfolio Projesi - Instruction Dosyasi

## Amac

Nuevo Softwarehouse AI Intern ilani icin optimize edilmis bir portfolyo projesi olusturmak.
Proje, AI agent workflow'lari, RAG pipeline, LLM karsilastirma ve local model deneyimlerini
Kubernetes uzerinde birlestiren kapsamli bir AI stack olacak.

Mevcut altyapi referansi: https://github.com/azize-devops/localai-rag-k3s

---

## Hedef Proje Yapisi

```
ai-stack-k8s/
├── README.md                          # AI-odakli proje tanitimi
├── .gitignore
├── deploy-all.sh
├── uninstall-all.sh
│
├── infrastructure/                    # Kubernetes deployment altyapisi
│   ├── namespace/
│   │   └── namespace.yaml
│   ├── localai/
│   │   ├── install.sh
│   │   ├── values.yaml
│   │   └── ingress.yaml
│   ├── ollama/                        # YENI: Ollama deployment
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── values.yaml
│   ├── qdrant/
│   │   ├── install.sh
│   │   └── values.yaml
│   └── anythingllm/
│       ├── install.sh
│       ├── values.yaml
│       └── ingress.yaml
│
├── agents/                            # YENI: AI Agent Workflow'lari
│   ├── README.md                      # Agent mimarisi aciklamasi
│   ├── requirements.txt
│   ├── crewai-research-agent/         # CrewAI multi-agent ornegi
│   │   ├── main.py
│   │   ├── agents.py                  # Agent tanimlari (researcher, writer, reviewer)
│   │   ├── tasks.py                   # Task tanimlari
│   │   └── config.yaml                # Agent konfigurasyonu
│   ├── langgraph-workflow/            # LangGraph state machine ornegi
│   │   ├── main.py
│   │   ├── graph.py                   # Workflow graph tanimi
│   │   ├── nodes.py                   # Node fonksiyonlari
│   │   └── state.py                   # State tanimlari
│   └── docker/
│       ├── Dockerfile
│       └── requirements.txt
│
├── rag-pipeline/                      # RAG Pipeline (mevcut + iyilestirilmis)
│   ├── README.md                      # RAG mimarisi aciklamasi
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── server.py                  # RAG API server
│   │   └── config.py
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── kustomization.yaml
│   └── pvc.yaml
│
├── llm-comparison/                    # YENI: LLM Karsilastirma
│   ├── README.md                      # Karsilastirma sonuclari ve analizler
│   ├── compare_models.py              # Farkli modelleri test eden script
│   ├── benchmark_results.json         # Benchmark sonuclari
│   └── metrics/
│       ├── response_quality.py        # Kalite degerlendirme
│       ├── latency.py                 # Hiz olcumu
│       └── memory_usage.py            # Bellek kullanimi
│
├── fine-tuning/                       # YENI: Fine-tuning Ornekleri
│   ├── README.md                      # Fine-tuning sureci dokumantasyonu
│   ├── prepare_dataset.py             # Dataset hazirlama
│   ├── lora_finetune.py               # LoRA fine-tuning script
│   ├── evaluate.py                    # Model degerlendirme
│   └── data/
│       └── sample_dataset.jsonl       # Ornek dataset
│
├── notebooks/                         # YENI: Demo Jupyter Notebook'lar
│   ├── 01_rag_pipeline_demo.ipynb     # RAG end-to-end demo
│   ├── 02_agent_workflow_demo.ipynb   # Agent workflow demo
│   ├── 03_llm_comparison.ipynb        # LLM karsilastirma notebook
│   └── 04_fine_tuning_demo.ipynb      # Fine-tuning demo
│
└── docs/
    ├── architecture.md                # Sistem mimarisi detayli aciklama
    └── images/
        └── architecture-diagram.png   # Mimari diyagram
```

---

## Modul Detaylari

### 1. agents/ - AI Agent Workflow'lari (EN YUKSEK ONCELIK)

**Neden:** Ilanin zorunlu gereksinimi. En az bir agent framework deneyimi isteniyor.

**CrewAI Research Agent:**
- 3 agent: Researcher (web aramasindan bilgi toplar), Writer (toplanan bilgiyi ozetler), Reviewer (kalite kontrol yapar)
- LocalAI backend kullanarak calismali
- Ornek use case: Bir konu hakkinda arastirma raporu olusturma

**LangGraph Workflow:**
- State machine tabanli workflow
- Ornek: Kullanici sorusu -> Siniflandirma -> RAG veya Direct LLM -> Cevap uretimi
- Conditional edge'ler ile dinamik routing

**Onemli:**
- Agent'lar LocalAI'nin OpenAI-compatible API'sini kullanmali (http://localai:8080/v1)
- Hem local model hem de harici API (OpenAI, Anthropic) ile calisabilmeli
- Her agent calismasinin log ve ciktisini kaydetmeli

### 2. llm-comparison/ - LLM Karsilastirma

**Neden:** Ilan "farkli LLM'leri test etme ve karsilastirma" istiyor.

**Karsilastirilacak Modeller:**
- LocalAI uzerinden: Mistral 7B, LLaMA 3 8B, Phi-3
- API uzerinden: GPT-4o-mini, Claude Haiku (opsiyonel)

**Metrikler:**
- Response kalitesi (ornek sorulara verdigi cevaplar)
- Latency (ilk token suresi, toplam sure)
- Bellek kullanimi
- Turkce dil destegi performansi

**Cikti:**
- benchmark_results.json icinde yapilandirilmis sonuclar
- README.md icinde ozet tablo ve analiz
- Notebook'ta gorsel grafikler

### 3. infrastructure/ollama/ - Ollama Deployment (YENI)

**Neden:** Ilanda spesifik olarak Ollama gecmektedir.

**Gereksinimler:**
- Kubernetes deployment manifest
- LocalAI ile yan yana calisabilmeli
- Ayni modelleri her iki platformda calistirip karsilastirma yapilabilmeli
- Pull model icin init container veya job

### 4. rag-pipeline/ - Gelistirilmis RAG

**Neden:** Mevcut rag-anything'in AI-odakli iyilestirilmesi.

**Iyilestirmeler:**
- Farkli embedding stratejileri (dense vs sparse vs hybrid)
- Chunk size ve overlap deneyimleri
- Re-ranking mekanizmasi
- Prompt template yonetimi
- Performans metrikleri ve loglama

### 5. fine-tuning/ - Fine-tuning Ornekleri

**Neden:** Ilan "basic fine-tuning experiments" istiyor.

**Kapsam:**
- QLoRA ile kucuk bir modeli fine-tune etme
- Custom dataset hazirlama pipeline'i
- Before/after karsilastirma
- Hugging Face PEFT kutuphanesi kullanimi

### 6. notebooks/ - Demo Notebook'lar

**Neden:** Projeyi canli gostermek ve teknik yetkinligi kanitlamak icin.

**Her notebook'ta olmasi gerekenler:**
- Markdown aciklamalar (ne yapildigini anlatan)
- Calistirilabilir kod hucreleri
- Cikti ornekleri (output hucreleri dolu olmali)
- Sonuc ve yorum

---

## README.md Yazim Kilavuzu

README su sekilde yapilandirilmali:

1. **Proje Basligi ve Tek Satirlik Aciklama**
   - "AI Stack on Kubernetes - Agent Workflows, RAG Pipeline & LLM Experimentation"

2. **Mimari Diyagram**
   - Tum bilesenlerin birbirine nasil baglandigini gosteren gorsel

3. **Ozellikler Listesi**
   - Multi-agent workflow (CrewAI + LangGraph)
   - RAG pipeline with vector search
   - LLM benchmarking across multiple models
   - Fine-tuning experiments
   - Kubernetes-native deployment

4. **Hizli Baslangic**
   - 3-5 adimda projeyi ayaga kaldirma

5. **Bilesenler Detay Tablosu**
   - Her bilesenin ne yaptigini, hangi teknolojiyi kullandigini gosteren tablo

6. **Demo / Screenshots**
   - Agent workflow ciktisi ornegi
   - RAG pipeline sorgu/cevap ornegi
   - LLM karsilastirma grafigi

7. **Teknoloji Stack'i**
   - Kubernetes, Helm, Python, CrewAI, LangGraph, LocalAI, Ollama, Qdrant, Docker

8. **Gelecek Planlar**
   - Phase 2 ozellikleri (gosterir ki proje aktif gelistiriliyor)

---

## Teknoloji Stack'i Ozeti

| Kategori | Teknolojiler |
|----------|-------------|
| Orkestrasyon | Kubernetes (K3s), Helm 3.x |
| LLM Backend | LocalAI, Ollama |
| Agent Framework | CrewAI, LangGraph |
| Vector DB | Qdrant |
| Fine-tuning | Hugging Face PEFT, QLoRA |
| Programlama | Python 3.11+, FastAPI |
| Container | Docker, Kustomize |
| RAG UI | AnythingLLM |

---

## Uygulama Sirasi

1. infrastructure/ - Mevcut K8s manifestlerini yeni yapiya tasi + Ollama ekle
2. agents/ - CrewAI ve LangGraph agent'larini olustur
3. rag-pipeline/ - Mevcut RAG kodunu iyilestir
4. llm-comparison/ - Benchmark scriptlerini yaz
5. fine-tuning/ - Fine-tuning pipeline olustur
6. notebooks/ - Demo notebook'lari hazirla
7. README.md - Proje tanitimini yaz
8. docs/ - Mimari dokumantasyon

---

## Notlar

- Tum Python kodlari icin requirements.txt dosyalari olusturulmali
- Her moduldeki README.md o modulun ne yaptigini aciklamali
- .gitignore guncellenmeli (model dosyalari, __pycache__, .env vb.)
- Hassas bilgiler (API key'ler) asla repo'ya eklenmemeli, .env.example kullanilmali
- Commit mesajlari aciklayici olmali
