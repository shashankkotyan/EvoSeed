import os
import cma
import pickle
import numpy as np

from tqdm import tqdm, trange
import more_itertools as itertools

import torch
from torchvision.transforms.functional import to_pil_image

import dnnlib
from robustbench import load_model


class Model:
    def __init__(
        self,
        device="cuda:0",
        generator_name="vp",
        classifier_name="Standard",
        threat="Linf",
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        self.device = device

        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        self.network_pkl = f"https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-{generator_name}.pkl"

        with dnnlib.util.open_url(self.network_pkl) as f:
            self.net = pickle.load(f)["ema"].to(device)

        self.vic_model = load_model(
            classifier_name, dataset="cifar10", threat_model=threat
        ).to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)

        self.sigma_min = max(self.sigma_min, self.net.sigma_min)
        self.sigma_max = min(self.sigma_max, self.net.sigma_max)

        self.step_indices = torch.arange(
            self.num_steps, dtype=torch.float64, device=self.device
        )
        self.t_steps = (
            self.sigma_max ** (1 / self.rho)
            + self.step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        self.t_steps = torch.cat(
            [self.net.round_sigma(self.t_steps), torch.zeros_like(self.t_steps[:1])]
        )

    @torch.no_grad()
    def generate_images(self, latents, labels):
        latents = latents.view(-1, 3, 32, 32)
        class_labels = torch.eye(self.net.label_dim, device=self.device)[labels]
        x_next = latents.to(torch.float64) * self.t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            x_cur = x_next

            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * self.S_noise * torch.randn_like(x_cur)

            denoised = self.net(x_hat, t_hat, class_labels)
            denoised = denoised.to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1:
                with torch.no_grad():
                    denoised = self.net(x_next, t_next, class_labels)

                denoised = denoised.to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        images = x_next
        images = (images * 127.5 + 128).clip(0, 255) / 255

        return images.float()

    @torch.no_grad()
    def classify_images(self, images):
        logits = self.vic_model(images)
        logits = self.softmax(logits)
        pred = logits.max(1, keepdim=True)[1][:, 0]

        return logits, pred

    @torch.no_grad()
    def evaluate(self, latents, labels):
        latents = torch.tensor(latents, device=self.device)
        labels = torch.tensor(labels, device=self.device)

        images = self.generate_images(latents, labels)
        logits, pred = self.classify_images(images)

        label_logits = torch.gather(logits, 1, labels[:, None])[:, 0]

        return {
            "latents": latents,
            "labels": labels,
            "images": images,
            "predictions": pred,
            "logits": logits,
            "fitness": label_logits.detach().cpu().numpy(),
        }


class EvoSeed:
    def __init__(
        self,
        generator_name="vp",
        classifier_name="Standard",
        threat="Linf",
        num_generations=100,
        eps=0.03,
        logdir="./log",
    ):
        self.num_generations = num_generations
        self.eps = eps

        self.model = Model(
            generator_name=generator_name,
            classifier_name=classifier_name,
            threat=threat,
        )

        self.latent_size = 3 * 32 * 32
        self.label_size = 10

        self.population_size = int(4 + 3 * np.log(self.latent_size))

        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)

        for i in range(self.label_size):
            os.makedirs(f"{logdir}/original/{i}", exist_ok=True)
            os.makedirs(f"{logdir}/adversarial/{i}", exist_ok=True)

        os.makedirs(f"{logdir}/data", exist_ok=True)

    def initialise(self, num):
        done = True

        while done:
            init = np.random.randn(self.latent_size)
            labels = np.random.randint(0, self.label_size, (1,))
            labels = np.repeat(labels, self.population_size, axis=0)

            opts = cma.CMAOptions()
            opts.set("bounds", [init - self.eps, init + self.eps])
            opts.set("verbose", -9)
            optimizer = cma.CMAEvolutionStrategy(list(init), 1.0, opts)

            d = self.model.evaluate(init[np.newaxis, ...], labels[np.newaxis, 0])
            done = d["labels"][0] != d["predictions"][0]

        to_pil_image(d["images"][0]).save(
            f"{self.logdir}/original/{d['labels'][0]}/{num:05d}_{d['predictions'][0]}.png"
        )

        os.makedirs(f"{self.logdir}/data/{num:05d}", exist_ok=True)
        torch.save(d, f"{self.logdir}/data/{num:05d}/init.pt")

        return labels, optimizer, 0, num

    def optimize(self, num_images):
        for i in trange(num_images):
            labels, optimizer, _, _ = self.initialise(i)

            for gen in range(self.num_generations):
                population = np.array(optimizer.ask())

                d = self.model.evaluate(population, labels)

                optimizer.tell(population, d["fitness"])
                torch.save(d, f"{self.logdir}/data/{i:05d}/{gen:03d}.pt")

                incorrect = torch.nonzero(d["predictions"] != d["labels"])[:, 0]

                if incorrect.shape[0] > 0:
                    for x in incorrect:
                        to_pil_image(d["images"][x, ...]).save(
                            f"{self.logdir}/adversarial/{d['labels'][x]}/{i:05d}_{d['predictions'][x]}_{gen + 1}_{x}.png"
                        )
                    break

                torch.cuda.empty_cache()


EvoSeed().optimize(1000)
